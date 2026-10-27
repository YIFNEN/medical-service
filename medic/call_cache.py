import json
import uuid
from pathlib import Path

from .ledger import AppendOnlyJsonlLedger, deterministic_id, sha256_digest
from .providers import GatewayResult, _LiveGateway
from .budget import BudgetLedger, request_bound
from .config import ModelRegistry


def cached_generate(ledger, gateway, *, scope, **kwargs):
    """Persist each successful paid boundary before starting the next call."""
    schema = kwargs["output_model"]
    key = deterministic_id("provider-call-v1", {
        "scope": scope, "provider": gateway.provider, "model": gateway.model,
        "stage": kwargs["stage"], "prompt": kwargs["prompt"],
        "schema": schema.model_json_schema(), "max_output_tokens": kwargs["max_output_tokens"],
        "images": [sha256_digest(Path(path).read_bytes()) for path in kwargs.get("image_paths", ())],
    })
    with ledger.key_lock("provider-call", key):
        existing = next((r for r in ledger.records()
                         if r["kind"] == "PROVIDER_CALL_COMPLETED" and r["natural_key"] == key), None)
        if existing:
            payload = existing["payload"]
            return GatewayResult(value=schema.model_validate_json(json.dumps(payload["value"])),
                                 **payload["metadata"])
        estimate = kwargs['estimate']
        if not estimate.approved:
            raise PermissionError('explicit cost approval is required before reservation')
        registry = getattr(gateway, 'registry', None) or ModelRegistry.default()
        if isinstance(gateway.inner, _LiveGateway) and estimate.budget_usd is None:
            raise PermissionError('live calls require an explicit budget_usd')
        budget = None
        if estimate.budget_usd is not None:
            budget = BudgetLedger(ledger, estimate.approval_id, estimate.budget_usd,
                                  registry.model_dump(mode='json'))
            bound = request_bound(registry, gateway.provider, kwargs['prompt'], schema.model_json_schema(),
                                  kwargs['max_output_tokens'], kwargs.get('image_paths', ()), estimate.image_tokens_bound)
            attempt = key + ':' + uuid.uuid4().hex
            budget.reserve(attempt, bound)
        # No hidden retries: failures keep their reservation. Resume is explicit.
        result = gateway.generate(**kwargs, _budget_reserved=budget is not None)
        ledger.append_once("PROVIDER_CALL_COMPLETED", key, {
            "stage": kwargs["stage"], "value": result.value.model_dump(mode="json"),
            "metadata": {"provider": result.provider, "model": result.model,
                         "response_id": result.response_id, "input_tokens": result.input_tokens,
                         "output_tokens": result.output_tokens},
        })
        if budget and (result.input_tokens or result.output_tokens):
            budget.settle(attempt, registry.call_record(kwargs['stage'], result).estimated_cost_usd)
        return result
