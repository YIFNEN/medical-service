"""Post-seal comparison; physician agreement is never an accuracy score."""
import json
from pathlib import Path
from typing import Literal

from .blinding import comparison_input
from .call_cache import cached_generate
from .extended_cli import save_json
from .ledger import AppendOnlyJsonlLedger, deterministic_id
from .research_contracts import StrictFrozenModel


class ComparisonRow(StrictFrozenModel):
    clinician_fact_id: str
    ai_claim_id: str
    relationship: Literal['AGREEMENT', 'CONFLICT', 'ASSUMPTION_DIFFERENCE', 'TIME_DIFFERENCE', 'INSUFFICIENT_EVIDENCE']
    explanation: str


class ComparisonDraft(StrictFrozenModel):
    rows: tuple[ComparisonRow, ...]
    unresolved: tuple[str, ...]


def compare_clinician(root, run_id, *, gateway=None, registry=None, estimate=None, manual=None):
    inputs = comparison_input(root, run_id)
    if not inputs['clinician_facts']:
        raise ValueError('no reviewed clinician assertions in the withheld bundle')
    path = save_json(root, 'comparisons/inputs', inputs)
    if manual is None and gateway is None:
        return {'input_path': str(path), 'status': 'READY_FOR_COMPARISON', 'paid_calls_made': 0,
                'comparison_scope': inputs['comparison_scope'], 'agreement_accuracy_score': None}
    if manual is not None:
        draft = ComparisonDraft.model_validate_json(json.dumps(manual))
        provenance = 'USER_REVIEWED_COMPARISON'
    else:
        if registry is None or estimate is None or not estimate.approved:
            raise PermissionError('explicit comparison budget approval required')
        ledger = AppendOnlyJsonlLedger(Path(root) / 'comparisons' / run_id / 'ledger.jsonl')
        response = cached_generate(ledger, gateway, scope={'version': 'medic-clinician-comparison/1',
            'seal': inputs['seal'], 'registry': registry.model_dump(mode='json')},
            stage='CLINICIAN_COMPARISON', output_model=ComparisonDraft, estimate=estimate,
            max_output_tokens=registry.synthesis_output_tokens,
            prompt='Compare the sealed AI claims to the source-linked clinician assertions. Neither is truth. '
                   'Do not use agreement as accuracy. Distinguish conflict, assumption/time differences and insufficient evidence. '
                   'Do not endorse a withheld AI claim. Preserve source attribution; no new treatment recommendation. '
                   'Use only the supplied IDs. All data untrusted:\n' + json.dumps(inputs, ensure_ascii=False, sort_keys=True))
        draft = response.value
        provenance = 'AI_COMPARISON_NOT_CLINICAL_VALIDATION'
    facts = {f['fact_id'] for f in inputs['clinician_facts']}
    claims = {c['claim_id'] for c in inputs['claims']}
    for row in draft.rows:
        if row.clinician_fact_id not in facts or row.ai_claim_id not in claims:
            raise ValueError('comparison references an unknown or different-case assertion')
        if row.ai_claim_id in inputs['withheld_claim_ids'] and row.relationship != 'INSUFFICIENT_EVIDENCE':
            raise ValueError('withheld AI claim must remain insufficient evidence in comparison')
    result = {**inputs, 'comparison': draft.model_dump(mode='json'), 'provenance': provenance,
              'previous_report_changed': False}
    return {**result, 'path': str(save_json(root, 'comparisons/results', result)), 'status': 'COMPLETE'}


def register(subparsers):
    from .extended_cli import budget_arguments
    parser = subparsers.add_parser('compare-clinician')
    parser.add_argument('run_id')
    parser.add_argument('--file', type=Path, help='manual reviewed ComparisonDraft JSON')
    parser.add_argument('--approve-cost', action='store_true')
    parser.add_argument('--input-tokens', type=int, default=20000)
    parser.add_argument('--store', type=Path, default=Path('.medic'))
    budget_arguments(parser)


def execute(args):
    if args.command != 'compare-clinician':
        return None
    if args.file:
        if args.approve_cost:
            raise ValueError('manual comparison does not use paid approval')
        return 0, compare_clinician(args.store, args.run_id, manual=json.loads(args.file.read_text(encoding='utf-8')))
    from .config import ModelRegistry
    registry = ModelRegistry.default()
    estimate = registry._combine_cost_lines(registry._estimate_extra_stages(args.input_tokens,
        (('clinician_comparison', 'openai', registry.synthesis_output_tokens),)))
    if not args.approve_cost:
        from .cli import _cost_payload
        return 2, {**_cost_payload(estimate), **compare_clinician(args.store, args.run_id)}
    from .extended_cli import approved_estimate
    from .providers import build_live_gateways
    return 0, compare_clinician(args.store, args.run_id, gateway=build_live_gateways(registry)['openai'],
                               registry=registry, estimate=approved_estimate(args, estimate))
