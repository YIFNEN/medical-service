from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .config import CostEstimate


T = TypeVar("T", bound=BaseModel)


class ProviderUnavailable(RuntimeError):
    pass


class TransientProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawProviderResponse:
    payload: dict[str, Any]
    response_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


@dataclass(frozen=True)
class GatewayResult(Generic[T]):
    value: T
    provider: str
    model: str
    response_id: str
    input_tokens: int
    output_tokens: int


class RawGateway(Protocol):
    provider: str
    model: str

    def generate_raw(
        self,
        *,
        stage: str,
        prompt: str,
        json_schema: dict[str, Any],
        max_output_tokens: int,
        image_paths: tuple[Path, ...] = (),
    ) -> RawProviderResponse: ...


class ApprovedGateway:
    def __init__(self, inner: RawGateway) -> None:
        self.inner = inner

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def generate(
        self,
        *,
        stage: str,
        prompt: str,
        output_model: type[T],
        estimate: CostEstimate,
        max_output_tokens: int = 24_000,
        image_paths: tuple[Path, ...] = (),
        _budget_reserved: bool = False,
    ) -> GatewayResult[T]:
        if not estimate.approved:
            raise PermissionError("explicit cost approval is required before provider calls")
        if isinstance(self.inner, _LiveGateway) and not _budget_reserved:
            raise PermissionError('live calls require the persistent budget reservation boundary')
        if isinstance(self.inner, _LiveGateway) and estimate.data_scope == 'PERSONAL' and not estimate.transmission_approved:
            raise PermissionError('separate personal-data transmission approval is required')
        response = self.inner.generate_raw(
            stage=stage,
            prompt=prompt,
            json_schema=output_model.model_json_schema(),
            max_output_tokens=max_output_tokens,
            image_paths=image_paths,
        )
        value = output_model.model_validate_json(json.dumps(response.payload, ensure_ascii=False))
        return GatewayResult(
            value=value,
            provider=self.inner.provider,
            model=response.model or self.inner.model,
            response_id=response.response_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


class RecordedGateway:
    def __init__(self, provider: str, model: str, responses: list[dict[str, Any]]) -> None:
        self.provider = provider
        self.model = model
        self._responses = list(responses)
        self.call_count = 0
        self.prompts: list[str] = []
        self.image_paths: list[tuple[Path, ...]] = []

    def generate_raw(
        self,
        *,
        stage: str,
        prompt: str,
        json_schema: dict[str, Any],
        max_output_tokens: int,
        image_paths: tuple[Path, ...] = (),
    ) -> RawProviderResponse:
        del stage, json_schema, max_output_tokens
        if not self._responses:
            raise RuntimeError(f"no recorded response remains for {self.provider}")
        self.call_count += 1
        self.prompts.append(prompt)
        self.image_paths.append(image_paths)
        payload = dict(self._responses.pop(0))
        explicit_response_id = payload.pop("_response_id", None)
        response_id = str(
            explicit_response_id
            or payload.get("provider_response_id")
            or payload.get("opinion_id")
            or payload.get("review_id")
            or f"recorded-{self.provider}-{self.call_count}"
        )
        return RawProviderResponse(payload=payload, response_id=response_id)


class _LiveGateway:
    provider: str

    def __init__(self, model: str, api_key_env: str, reasoning: str | None = None) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.reasoning = reasoning

    def ensure_available(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderUnavailable(
                f"{self.provider} requires environment variable {self.api_key_env}"
            )
        return key

    @staticmethod
    def request_once(operation):
        return operation()

    @staticmethod
    def retry_transient(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception as exc:
                name = type(exc).__name__.casefold()
                status = getattr(exc, "status_code", getattr(exc, "code", None))
                if status not in {429, 500, 502, 503, 504} and not any(token in name for token in ("ratelimit", "connection", "timeout")):
                    raise
                if attempt == 2:
                    raise TransientProviderError(f"provider transient failure: {type(exc).__name__}") from exc
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def _image_data(path: Path) -> tuple[str, str]:
    path = Path(path)
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid provider image: {path}") from exc
    mime = Image.MIME.get(str(image_format)) or mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
        raise ValueError(f"unsupported provider image type: {path}")
    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


def _strict_schema(schema):
    schema = json.loads(json.dumps(schema))
    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(schema)
    return schema


class OpenAIGateway(_LiveGateway):
    provider = "openai"

    def generate_raw(self, *, stage, prompt, json_schema, max_output_tokens, image_paths=()):
        del stage
        key = self.ensure_available()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderUnavailable("install the 'openai' package") from exc
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path in image_paths:
            mime, encoded = _image_data(path)
            content.append({"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "high"})
        client = OpenAI(api_key=key, max_retries=0, timeout=180.0)
        response = self.request_once(lambda: client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
            reasoning={"effort": self.reasoning or "max"},
            text={"format": {"type": "json_schema", "name": "medic_structured_output", "schema": _strict_schema(json_schema), "strict": True}},
        ))
        if response.status != "completed":
            raise RuntimeError(f"OpenAI response {response.id}: {response.status}")
        if any(getattr(part, "type", None) == "refusal" for item in response.output
               for part in getattr(item, "content", ())):
            raise RuntimeError(f"OpenAI response {response.id}: refusal")
        usage = getattr(response, "usage", None)
        return RawProviderResponse(
            payload=json.loads(response.output_text), response_id=response.id,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            model=response.model,
        )


class AnthropicGateway(_LiveGateway):
    provider = "anthropic"

    def generate_raw(self, *, stage, prompt, json_schema, max_output_tokens, image_paths=()):
        del stage
        key = self.ensure_available()
        try:
            from anthropic import Anthropic, transform_schema
        except ImportError as exc:
            raise ProviderUnavailable("install the 'anthropic' package") from exc
        content: list[dict[str, Any]] = []
        for path in image_paths:
            mime, encoded = _image_data(path)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}})
        content.append({"type": "text", "text": prompt})
        client = Anthropic(api_key=key, max_retries=0, timeout=180.0)
        response = self.request_once(lambda: client.messages.create(
            model=self.model, max_tokens=max_output_tokens,
            messages=[{"role": "user", "content": content}],
            thinking={"type": self.reasoning or "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": transform_schema(json_schema)}},
        ))
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"Anthropic response {response.id}: {response.stop_reason}")
        text = next(block.text for block in response.content if block.type == "text")
        return RawProviderResponse(
            payload=json.loads(text), response_id=response.id,
            input_tokens=int(getattr(response.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response.usage, "output_tokens", 0) or 0),
            model=response.model,
        )


class GeminiGateway(_LiveGateway):
    provider = "google"

    def generate_raw(self, *, stage, prompt, json_schema, max_output_tokens, image_paths=()):
        del stage
        key = self.ensure_available()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderUnavailable("install the 'google-genai' package") from exc
        contents: list[Any] = [prompt]
        for path in image_paths:
            mime, _encoded = _image_data(path)
            contents.append(types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mime))
        response = self.request_once(lambda: genai.Client(api_key=key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))).models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=json_schema,
                max_output_tokens=max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_level=(self.reasoning or "high").upper()),
            ),
        ))
        if not response.candidates:
            raise RuntimeError("Gemini response blocked or empty")
        finish = str(response.candidates[0].finish_reason).split(".")[-1]
        if finish != "STOP":
            raise RuntimeError(f"Gemini response stopped: {finish}")
        usage = response.usage_metadata
        return RawProviderResponse(
            payload=json.loads(response.text),
            response_id=str(getattr(response, "response_id", "google-response")),
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0)
                          + int(getattr(usage, "thoughts_token_count", 0) or 0),
            model=getattr(response, "model_version", None),
        )


def build_live_gateways(registry) -> dict[str, ApprovedGateway]:
    classes = {"openai": OpenAIGateway, "anthropic": AnthropicGateway, "google": GeminiGateway}
    gateways = {
        item.provider: ApprovedGateway(classes[item.provider](item.model, item.api_key_env, item.reasoning))
        for item in registry.providers
    }
    for gateway in gateways.values():
        gateway.registry = registry
    return gateways
