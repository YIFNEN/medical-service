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
    if set(gateways) != {item.provider for item in registry.providers}:
        raise ValueError('smoke gateways must match the configured providers')
    ledger = AppendOnlyJsonlLedger(Path(root) / 'provider-checks' / run_id / 'ledger.jsonl')
    ledger.append_once('SMOKE_CREATED', run_id, {'registry': registry.model_dump(mode='json'),
        'models': {key: value.model for key, value in gateways.items()},
        'estimate': estimate.model_dump(mode='json'), 'version': 'medic-smoke/1'})
    result = {'run_id': run_id, 'input_scope': 'SYNTHETIC_ONLY', 'providers': {},
              'status': 'COMPLETE', 'clinical_validation': 'NOT_VALIDATED'}
    if estimate.backend == 'codex':
        result['backend'] = 'codex'
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


class ImageSmokeReply(StrictFrozenModel):
    square_color: Literal['blue', 'red', 'green', 'white']


def image_smoke(root, identifier, registry, estimate, gateway):
    from secrets import choice
    from tempfile import TemporaryDirectory
    from PIL import Image, ImageDraw
    from .extended_cli import run_id as validate_run_id

    if (estimate.backend != 'codex' or estimate.codex_calls != 1
            or estimate.data_scope != 'SYNTHETIC' or not estimate.approved):
        raise PermissionError('synthetic Codex image smoke requires one approved attempt')
    ledger = AppendOnlyJsonlLedger(Path(root) / 'provider-checks' / validate_run_id(identifier) / 'ledger.jsonl')
    pin = {'estimate': estimate.model_dump(mode='json'), 'model': gateway.model}
    with ledger.key_lock('image-smoke', identifier):
        saved = next((r['payload'] for r in ledger.records() if r['kind'] == 'IMAGE_SMOKE_CREATED'), None)
        if saved is None:
            pin['color'] = choice(('blue', 'red', 'green'))
        elif 'color' in saved:
            pin['color'] = saved['color']
        ledger.append_once('IMAGE_SMOKE_CREATED', identifier, pin)
    color = pin.get('color', 'blue')
    with TemporaryDirectory(prefix='medic-image-smoke-') as directory:
        path = Path(directory) / 'synthetic.png'
        image = Image.new('RGB', (256, 256), 'white')
        ImageDraw.Draw(image).rectangle((64, 64, 191, 191), fill=color)
        image.save(path)
        response = cached_generate(ledger, gateway, scope={'version': 'image-smoke/1'},
            estimate=estimate, stage='SYNTHETIC_IMAGE_SMOKE',
            prompt='Read the attached synthetic image. Return the color of the center square.',
            output_model=ImageSmokeReply, max_output_tokens=1024, image_paths=(path,))
    if response.value.square_color != color:
        raise ValueError('synthetic image color did not match')
    return {'status': 'PASSED', 'input_scope': 'SYNTHETIC_ONLY',
            'call': registry.call_record('SYNTHETIC_IMAGE_SMOKE', response).model_dump(mode='json')}


def register(subparsers):
    from .extended_cli import budget_arguments
    parser = subparsers.add_parser('providers').add_subparsers(dest='provider_action', required=True).add_parser('smoke')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--approve-cost', action='store_true')
    parser.add_argument('--input-tokens', type=int, default=1000)
    parser.add_argument('--synthetic-image', action='store_true')
    budget_arguments(parser)
    parser.add_argument('--store', type=Path, default=Path('.medic'))


def execute(args):
    if args.command != 'providers':
        return None
    from .config import ModelRegistry
    from .extended_cli import approved_estimate, execution_estimate, execution_requested
    from .providers import build_live_gateways
    registry = ModelRegistry.for_cli()
    if args.synthetic_image and args.backend != 'codex':
        raise ValueError('--synthetic-image requires --backend codex')
    estimate = registry._combine_cost_lines(registry._estimate_extra_stages(args.input_tokens,
        tuple(('provider_smoke', p.provider, 1024) for p in registry.providers)))
    estimate = execution_estimate(args, estimate.model_copy(update={'data_scope': 'SYNTHETIC'}))
    if not execution_requested(args):
        from .cli import _cost_payload
        return 2, {**_cost_payload(estimate), 'input_scope': 'SYNTHETIC_ONLY', 'run_id': args.run_id}
    if args.synthetic_image:
        gateway = build_live_gateways(registry, backend='codex')['openai']
        return 0, image_smoke(args.store, args.run_id, registry, approved_estimate(args, estimate), gateway)
    result = smoke(args.store, args.run_id, build_live_gateways(registry, backend=estimate.backend),
                   registry, approved_estimate(args, estimate))
    return (0 if result['status'] == 'COMPLETE' else 2), result
