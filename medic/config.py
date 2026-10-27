from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from pydantic import Field

from .research_contracts import ProviderCallRecord, StrictFrozenModel


class ProviderModel(StrictFrozenModel):
    provider: str
    model: str
    api_key_env: str
    input_usd_per_million: float = Field(gt=0)
    output_usd_per_million: float = Field(gt=0)
    reasoning: str

    @property
    def key_available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))


class CostLine(StrictFrozenModel):
    stage: str
    provider: str
    calls: int
    max_output_tokens: int
    estimated_usd: float


class CostEstimate(StrictFrozenModel):
    pricing_as_of: date
    lines: tuple[CostLine, ...]
    call_count: int
    estimated_usd: float
    approved: bool = False
    budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    approval_id: str = 'default'
    image_tokens_bound: int | None = Field(default=None, gt=0)

    def approve(self) -> "CostEstimate":
        return self.model_copy(update={"approved": True})


class ModelRegistry(StrictFrozenModel):
    pricing_as_of: date
    providers: tuple[ProviderModel, ...]
    analyst_output_tokens: int = 24_000
    peer_review_output_tokens: int = 8_000
    revision_output_tokens: int = 12_000
    synthesis_output_tokens: int = 16_000
    synthesis_review_output_tokens: int = 6_000

    @classmethod
    def default(cls) -> "ModelRegistry":
        if os.environ.get("MEDIC_MODEL_REGISTRY"):
            return cls.model_validate_json(Path(os.environ["MEDIC_MODEL_REGISTRY"]).read_bytes())
        defaults = {"OPENAI": "gpt-5.6-sol", "ANTHROPIC": "claude-opus-4-7", "GEMINI": "gemini-3.7-flash"}
        for name, model in defaults.items():
            if os.environ.get(f"MEDIC_{name}_MODEL", model) != model:
                raise ValueError("custom model requires explicit model-specific pricing via a ModelRegistry; default prices cannot be reused")
        return cls(
            pricing_as_of=date(2026, 9, 19),
            providers=(
                ProviderModel(
                    provider="openai",
                    model=os.environ.get("MEDIC_OPENAI_MODEL", "gpt-5.6-sol"),
                    api_key_env="OPENAI_API_KEY",
                    input_usd_per_million=4.0,
                    output_usd_per_million=20.0,
                    reasoning="max",
                ),
                ProviderModel(
                    provider="anthropic",
                    model=os.environ.get("MEDIC_ANTHROPIC_MODEL", "claude-opus-4-7"),
                    api_key_env="ANTHROPIC_API_KEY",
                    input_usd_per_million=5.0,
                    output_usd_per_million=25.0,
                    reasoning="adaptive",
                ),
                ProviderModel(
                    provider="google",
                    model=os.environ.get("MEDIC_GEMINI_MODEL", "gemini-3.7-flash"),
                    api_key_env="GEMINI_API_KEY",
                    input_usd_per_million=0.75,
                    output_usd_per_million=3.75,
                    reasoning="high",
                ),
            ),
        )

    def call_record(self, stage: str, response) -> ProviderCallRecord:
        pricing = next(item for item in self.providers if item.provider == response.provider)
        cost = (response.input_tokens * pricing.input_usd_per_million
                + response.output_tokens * pricing.output_usd_per_million) / 1_000_000
        return ProviderCallRecord(
            stage=stage, provider=response.provider, model=response.model,
            provider_response_id=response.response_id,
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
            estimated_cost_usd=round(cost, 6),
        )

    def estimate_tribunal_cost(self, input_tokens_per_call: int) -> CostEstimate:
        stages = (
            ("analyst", 1, self.analyst_output_tokens),
            ("peer_review", 1, self.peer_review_output_tokens),
            ("revision", 1, self.revision_output_tokens),
            ("synthesis_review", 1, self.synthesis_review_output_tokens),
        )
        lines: list[CostLine] = []
        for provider in self.providers:
            for stage, calls, output_tokens in stages:
                estimated = calls * (
                    input_tokens_per_call * provider.input_usd_per_million
                    + output_tokens * provider.output_usd_per_million
                ) / 1_000_000
                lines.append(
                    CostLine(
                        stage=stage,
                        provider=provider.provider,
                        calls=calls,
                        max_output_tokens=output_tokens,
                        estimated_usd=round(estimated, 6),
                    )
                )
        synthesis_provider = self.providers[0]
        synthesis_cost = (
            input_tokens_per_call * synthesis_provider.input_usd_per_million
            + self.synthesis_output_tokens * synthesis_provider.output_usd_per_million
        ) / 1_000_000
        lines.append(
            CostLine(
                stage="synthesis",
                provider=synthesis_provider.provider,
                calls=1,
                max_output_tokens=self.synthesis_output_tokens,
                estimated_usd=round(synthesis_cost, 6),
            )
        )
        return CostEstimate(
            pricing_as_of=self.pricing_as_of,
            lines=tuple(lines),
            call_count=sum(line.calls for line in lines),
            estimated_usd=round(sum(line.estimated_usd for line in lines), 6),
        )
    def _estimate_extra_stages(
        self,
        input_tokens_per_call: int,
        stages: tuple[tuple[str, str, int], ...],
    ) -> tuple[CostLine, ...]:
        by_provider = {item.provider: item for item in self.providers}
        lines: list[CostLine] = []
        for stage, provider_name, output_tokens in stages:
            provider = by_provider[provider_name]
            estimated = (
                input_tokens_per_call * provider.input_usd_per_million
                + output_tokens * provider.output_usd_per_million
            ) / 1_000_000
            lines.append(
                CostLine(
                    stage=stage,
                    provider=provider_name,
                    calls=1,
                    max_output_tokens=output_tokens,
                    estimated_usd=round(estimated, 6),
                )
            )
        return tuple(lines)

    def _combine_cost_lines(self, lines: tuple[CostLine, ...]) -> CostEstimate:
        return CostEstimate(
            pricing_as_of=self.pricing_as_of,
            lines=lines,
            call_count=sum(line.calls for line in lines),
            estimated_usd=round(sum(line.estimated_usd for line in lines), 6),
        )

    def estimate_extraction_cost(self, input_tokens_per_call: int, batch_count: int = 1) -> CostEstimate:
        if batch_count < 0:
            raise ValueError("batch_count cannot be negative")
        lines = self._estimate_extra_stages(
            input_tokens_per_call,
            (
                ("document_extraction", "openai", self.analyst_output_tokens),
                ("document_extraction", "google", self.analyst_output_tokens),
                ("document_reconciliation", "anthropic", self.revision_output_tokens),
            ),
        )
        return self._combine_cost_lines(tuple(line.model_copy(update={
            "calls": line.calls * batch_count,
            "estimated_usd": round(line.estimated_usd * batch_count, 6),
        }) for line in lines))

    def estimate_deliberation_cost(self, input_tokens_per_call: int, evidence_limit: int = 0) -> CostEstimate:
        tribunal = self.estimate_tribunal_cost(input_tokens_per_call)
        planning = self._estimate_extra_stages(
            input_tokens_per_call,
            tuple(
                ("literature_planning", provider, self.peer_review_output_tokens)
                for provider in ("openai", "anthropic", "google")
            ),
        )
        appraisal = self._estimate_extra_stages(input_tokens_per_call,
            tuple(("evidence_appraisal", "anthropic", 8000) for _ in range(evidence_limit)))
        return self._combine_cost_lines(tribunal.lines + planning + appraisal)

    def estimate_full_workflow_cost(self, input_tokens_per_call: int) -> CostEstimate:
        extraction = self.estimate_extraction_cost(input_tokens_per_call)
        deliberation = self.estimate_deliberation_cost(input_tokens_per_call)
        return self._combine_cost_lines(extraction.lines + deliberation.lines)

    def estimate_research_cost(self, input_tokens_per_call, evidence_limit=20, fulltext_limit=5):
        if evidence_limit < 1 or fulltext_limit < 0:
            raise ValueError('invalid literature limits')
        base = self.estimate_deliberation_cost(input_tokens_per_call, evidence_limit)
        stages = (('decision_issues', 'anthropic', 4000),)
        stages += tuple(('screening', 'anthropic', 6000) for _ in range(20))
        stages += tuple(('fulltext_chunk', 'anthropic', 8000) for _ in range(fulltext_limit * 12))
        return self._combine_cost_lines(base.lines + self._estimate_extra_stages(input_tokens_per_call, stages))
