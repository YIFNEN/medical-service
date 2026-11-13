import json
import uuid
from pathlib import Path

from .ledger import AppendOnlyJsonlLedger, deterministic_id, sha256_digest
from .providers import GatewayResult, _LiveGateway
from .budget import BudgetLedger, request_bound
from .config import ModelRegistry


def cached_generate(ledger, gateway, *, scope, validate_result=None, **kwargs):
    """Persist each successful paid boundary before starting the next call."""
    if ledger.path.parent.parent.name == 'runs':
        from .blinding import assert_no_leak
        assert_no_leak(ledger.path.parent.parent.parent, ledger.path.parent.name, kwargs['prompt'])
    schema = kwargs["output_model"]
    estimate = kwargs['estimate']
    backend = gateway.backend
    if estimate.backend != backend:
        raise PermissionError('execution backend must match the approved estimate')
    key = deterministic_id("provider-call-v1", {
        **({'backend': backend} if backend != 'api' else {}),
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
            value = schema.model_validate_json(json.dumps(payload["value"]))
            if validate_result:
                validate_result(value)
            return GatewayResult(value=value, **payload["metadata"])
        if not estimate.approved:
            raise PermissionError('explicit cost approval is required before reservation')
        registry = getattr(gateway, 'registry', None) or ModelRegistry.default()
        if (isinstance(gateway.inner, _LiveGateway) or backend == 'codex') and estimate.data_scope == 'PERSONAL' and not estimate.transmission_approved:
            raise PermissionError('separate personal-data transmission approval is required before reservation')
        if isinstance(gateway.inner, _LiveGateway) and estimate.budget_usd is None:
            raise PermissionError('live calls require an explicit budget_usd')
        budget = None
        if backend == 'codex':
            if not estimate.codex_calls or estimate.budget_usd is not None:
                raise ValueError('Codex execution requires --codex-calls and no USD budget')
            approval = estimate.approval_id
            with ledger.key_lock('codex-attempts', approval):
                ledger.append_once('CODEX_LIMIT_APPROVED', approval,
                                   {'limit': estimate.codex_calls, 'model': gateway.model})
                used = sum(r['kind'] == 'CODEX_ATTEMPT_RESERVED' and r['payload']['approval_id'] == approval
                           for r in ledger.records())
                if used >= estimate.codex_calls:
                    raise RuntimeError('Codex attempt limit reached')
                ledger.append_once('CODEX_ATTEMPT_RESERVED', key + ':' + uuid.uuid4().hex,
                                   {'approval_id': approval})
        if estimate.budget_usd is not None:
            budget = BudgetLedger(ledger, estimate.approval_id, estimate.budget_usd,
                                  registry.model_dump(mode='json'))
            bound = request_bound(registry, gateway.provider, kwargs['prompt'], schema.model_json_schema(),
                                  kwargs['max_output_tokens'], kwargs.get('image_paths', ()), estimate.image_tokens_bound)
            attempt = key + ':' + uuid.uuid4().hex
            budget.reserve(attempt, bound)
        # No hidden retries: failures keep their reservation. Resume is explicit.
        result = gateway.generate(**kwargs, _budget_reserved=budget is not None or backend == 'codex')
        if validate_result:
            validate_result(result.value)
        ledger.append_once("PROVIDER_CALL_COMPLETED", key, {
            "stage": kwargs["stage"], "value": result.value.model_dump(mode="json"),
            "metadata": {"provider": result.provider, "model": result.model,
                         **({'backend': backend} if backend != 'api' else {}),
                         "response_id": result.response_id, "input_tokens": result.input_tokens,
                         "output_tokens": result.output_tokens},
        })
        if budget and (result.input_tokens or result.output_tokens):
            budget.settle(attempt, registry.call_record(kwargs['stage'], result).estimated_cost_usd)
        return result
