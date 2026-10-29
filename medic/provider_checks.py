"""Synthetic, explicitly approved provider connectivity checks with resumable calls."""
from pathlib import Path
from typing import Literal

from .call_cache import cached_generate
from .ledger import AppendOnlyJsonlLedger
from .research_contracts import StrictFrozenModel


class SmokeReply(StrictFrozenModel):
    echo: Literal['MEDIC_SYNTHETIC_SMOKE']
    structured: Literal[True]


def smoke(root, run_id, gateways, registry, estimate):
    from .extended_cli import run_id as validate_run_id
    validate_run_id(run_id)
    if not estimate.approved:
        raise PermissionError('explicit cost approval required for provider smoke')
    if set(gateways) != {'openai', 'anthropic', 'google'}:
        raise ValueError('all three providers are required')
    ledger = AppendOnlyJsonlLedger(Path(root) / 'provider-checks' / run_id / 'ledger.jsonl')
    ledger.append_once('SMOKE_CREATED', run_id, {'registry': registry.model_dump(mode='json'),
        'models': {key: value.model for key, value in gateways.items()},
        'estimate': estimate.model_dump(mode='json'), 'version': 'medic-smoke/1'})
    result = {'run_id': run_id, 'input_scope': 'SYNTHETIC_ONLY', 'providers': {},
              'status': 'COMPLETE', 'clinical_validation': 'NOT_VALIDATED'}
    for name, gateway in gateways.items():
        try:
            response = cached_generate(ledger, gateway,
                scope={'version': 'medic-smoke/1', 'registry': registry.model_dump(mode='json')},
                stage='PROVIDER_SMOKE', prompt='Synthetic connectivity check. Return echo MEDIC_SYNTHETIC_SMOKE and structured true.',
                output_model=SmokeReply, estimate=estimate, max_output_tokens=1024)
            result['providers'][name] = {'status': 'PASSED', 'configured_model': gateway.model,
                'returned_model': response.model, 'response_id': response.response_id,
                'input_tokens': response.input_tokens, 'output_tokens': response.output_tokens}
        except Exception as exc:
            # SDK exception text may contain request data, identifiers or credentials.
            result['providers'][name] = {'status': 'FAILED', 'error_type': type(exc).__name__}
            result['status'] = 'INCOMPLETE'
    return result


def register(subparsers):
    from .extended_cli import budget_arguments
    parser = subparsers.add_parser('providers').add_subparsers(dest='provider_action', required=True).add_parser('smoke')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--approve-cost', action='store_true')
    parser.add_argument('--input-tokens', type=int, default=1000)
    budget_arguments(parser)
    parser.add_argument('--store', type=Path, default=Path('.medic'))


def execute(args):
    if args.command != 'providers':
        return None
    from .config import ModelRegistry
    from .extended_cli import approved_estimate
    from .providers import build_live_gateways
    registry = ModelRegistry.default()
    estimate = registry._combine_cost_lines(registry._estimate_extra_stages(args.input_tokens,
        tuple(('provider_smoke', p.provider, 1024) for p in registry.providers)))
    estimate = estimate.model_copy(update={'data_scope': 'SYNTHETIC'})
    if not args.approve_cost:
        from .cli import _cost_payload
        return 2, {**_cost_payload(estimate), 'input_scope': 'SYNTHETIC_ONLY', 'run_id': args.run_id}
    if args.budget_usd is None:
        raise ValueError('paid calls require --budget-usd')
    result = smoke(args.store, args.run_id, build_live_gateways(registry), registry, approved_estimate(args, estimate))
    return (0 if result['status'] == 'COMPLETE' else 2), result
