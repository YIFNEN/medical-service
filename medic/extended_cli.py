"""Local inventory/review commands and explicit literature stage entry points."""
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, sha256_digest
from .config import CostEstimate, ModelRegistry


def run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('invalid run identifier')
    return value


def budget_arguments(parser):
    parser.add_argument('--budget-usd', type=float)
    parser.add_argument('--approval-id', default='default')
    parser.add_argument('--image-tokens-bound', type=int)


def approved_estimate(args, estimate):
    if args.budget_usd is None:
        raise ValueError('paid calls require --budget-usd and --approve-cost')
    return CostEstimate.model_validate({**estimate.model_dump(), 'approved': True,
        'budget_usd': args.budget_usd, 'approval_id': args.approval_id,
        'image_tokens_bound': args.image_tokens_bound})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(root, category, value):
    content = canonical_json_bytes(value)
    directory = Path(root) / category
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (sha256_digest(content).split(':')[1] + '.json')
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('stored artifact content mismatch')
    else:
        with path.open('xb') as stream:
            stream.write(content)
    return path


def register(subparsers):
    def store(parser):
        parser.add_argument('--store', type=Path, default=Path('.medic'))
    inventory = subparsers.add_parser('inventory')
    inventory.add_argument('source', type=Path)
    inventory.add_argument('--max-member-mb', type=int, default=512)
    inventory.add_argument('--max-total-mb', type=int, default=8192)
    store(inventory)
    roles = subparsers.add_parser('inventory-review').add_subparsers(dest='action', required=True)
    for action in ('status', 'assign', 'withdraw'):
        parser = roles.add_parser(action)
        parser.add_argument('--inventory', type=Path, required=True)
        store(parser)
        if action != 'status':
            parser.add_argument('--record-id', required=True)
            parser.add_argument('--reviewer', required=True)
            parser.add_argument('--reason', required=True)
        if action == 'assign':
            parser.add_argument('--role', choices=('PATIENT_RECORD', 'LITERATURE', 'OTHER'), required=True)
            parser.add_argument('--case')
            parser.add_argument('--citation')
    dicom = subparsers.add_parser('dicom').add_subparsers(dest='action', required=True)
    prepare = dicom.add_parser('prepare-slicer-test')
    store(prepare)
    source_test = dicom.add_parser('prepare-slicer-source-test')
    source_test.add_argument('--manifest', type=Path, required=True)
    source_test.add_argument('--series')
    store(source_test)
    inspect = dicom.add_parser('inspect')
    inspect.add_argument('--inventory', type=Path, required=True)
    store(inspect)
    export = dicom.add_parser('export')
    export.add_argument('--manifest', type=Path, required=True)
    export.add_argument('--series', required=True)
    store(export)
    review = dicom.add_parser('import-review')
    review.add_argument('--manifest', type=Path, required=True)
    review.add_argument('--review', type=Path, required=True)
    review.add_argument('--case', required=True)
    review.add_argument('--reviewer', required=True)
    review.add_argument('--review-status', choices=('USER_REVIEWED', 'EXPERT_REVIEWED'), default='USER_REVIEWED')
    review.add_argument('--replaces-review-id')
    review.add_argument('--amendment-reason')
    store(review)
    withdraw = dicom.add_parser('withdraw-review')
    withdraw.add_argument('--case', required=True)
    withdraw.add_argument('--review-id', required=True)
    withdraw.add_argument('--reviewer', required=True)
    withdraw.add_argument('--reason', required=True)
    store(withdraw)
    literature = subparsers.add_parser('literature').add_subparsers(dest='action', required=True)
    for action in ('screen', 'fetch', 'appraise'):
        parser = literature.add_parser(action)
        parser.add_argument('--cards', type=Path, required=True)
        parser.add_argument('--context', type=Path)
        parser.add_argument('--issues', type=Path)
        parser.add_argument('--case', default='local-research')
        parser.add_argument('--question')
        parser.add_argument('--run-id', default='literature-stages')
        parser.add_argument('--local-papers', type=Path)
        parser.add_argument('--fulltext-limit', type=int, default=5)
        if action == 'fetch':
            parser.add_argument('--retry-unavailable', action='store_true')
        parser.add_argument('--approve-cost', action='store_true')
        budget_arguments(parser)
        store(parser)
    query = literature.add_parser('search-plan')
    query.add_argument('run_id')
    store(query)
    compare = subparsers.add_parser('compare-runs')
    compare.add_argument('before')
    compare.add_argument('after')
    store(compare)


def execute(args):
    root = Path(args.store)
    if args.command == 'inventory-review':
        from .material_roles import role_view, review_role
        inventory = read_json(args.inventory)
        inventory_path = save_json(root, 'inventory', inventory)
        ledger = AppendOnlyJsonlLedger(root / 'ledger.jsonl')
        if args.action == 'status':
            view = role_view(inventory, ledger)
            return 0, {'path': str(save_json(root, 'inventory/reviews', view)),
                       'inventory_path': str(inventory_path),
                       'record_count': len(view['records']),
                       'unknown_count': sum(r['role'] == 'UNKNOWN' for r in view['records'])}
        event_id = review_role(inventory, ledger, args.record_id,
            role=args.role if args.action == 'assign' else 'UNKNOWN', reviewer=args.reviewer,
            reason=args.reason, case_id=getattr(args, 'case', None), citation=getattr(args, 'citation', None))
        return 0, {'event_id': event_id, 'record_id': args.record_id, 'inventory_path': str(inventory_path),
                   'role': args.role if args.action == 'assign' else 'UNKNOWN'}
    if args.command == 'inventory':
        from .inventory import scan, Limits
        result = scan(args.source, limits=Limits(max_member_bytes=args.max_member_mb * 1024**2,
                                                max_total_bytes=args.max_total_mb * 1024**2))
        path = save_json(root, 'inventory', result)
        return 0, {'path': str(path), 'record_count': len(result['records']), 'truncated': result['truncated']}
    if args.command == 'dicom':
        if args.action == 'prepare-slicer-test':
            from .slicer_validation import prepare_validation
            return 0, prepare_validation(root)
        if args.action == 'prepare-slicer-source-test':
            from .slicer_validation import prepare_source_validation
            return 0, prepare_source_validation(root, args.manifest, args.series)
        if args.action == 'withdraw-review':
            if not args.reason.strip() or not args.reviewer.strip():
                raise ValueError('reviewer and withdrawal reason must be nonempty')
            ledger = AppendOnlyJsonlLedger(root / 'ledger.jsonl')
            key = args.case + ':' + args.review_id
            with ledger.key_lock('imaging-review', key):
                records = ledger.records()
                if not any(r['kind'] == 'IMAGING_REVIEW_IMPORTED'
                           and r['payload'].get('case_id') == args.case
                           and r['payload'].get('review_id') == args.review_id for r in records):
                    raise ValueError('unknown imaging review for case')
                payload = {'case_id': args.case, 'review_id': args.review_id,
                           'reviewer': args.reviewer.strip(), 'reason': args.reason.strip()}
                previous = next((r for r in records if r['kind'] == 'IMAGING_REVIEW_WITHDRAWN'
                                 and r['natural_key'] == key), None)
                if previous:
                    if any(previous['payload'].get(k) != v for k, v in payload.items()):
                        raise ValueError('withdrawal already recorded with different details')
                else:
                    ledger.append_once('IMAGING_REVIEW_WITHDRAWN', key,
                                       {**payload, 'withdrawn_at': datetime.now(UTC).isoformat()})
            return 0, {'review_id': args.review_id, 'status': 'WITHDRAWN',
                       'past_reports_changed': False}
        from .imaging import inspect_inventory
        from .slicer_review import review_template, import_review
        if args.action == 'inspect':
            manifest = inspect_inventory(read_json(args.inventory), root / 'imaging' / 'originals')
            path = save_json(root, 'imaging/manifests', manifest)
            return 0, {'path': str(path), 'series_count': len(manifest['series']),
                       'measurement_eligible_count': sum(s['measurement_eligible'] for s in manifest['series']),
                       'instance_count': len(manifest['instances']), 'conflict_count': len(manifest['conflicts'])}
        manifest = read_json(args.manifest)
        if args.action == 'export':
            review = review_template(manifest, args.series)
            path = save_json(root, 'imaging/templates', review)
            script = Path(__file__).resolve().parent.parent / 'scripts' / 'slicer_review.py'
            return 0, {'path': str(path), 'manifest': str(args.manifest.resolve()), 'slicer_script': str(script),
                       'instruction': 'Use load_series and export_review in 3D Slicer; then dicom import-review.',
                       'clinical_validation': 'NOT_VALIDATED'}
        result = import_review(read_json(args.review), manifest, review_dir=args.review.parent)
        result.update(reviewer=args.reviewer, review_status=args.review_status,
                      review_status_provenance='USER_DECLARED_NOT_CREDENTIAL_VERIFIED')
        path = save_json(root, 'imaging/reviews', result)
        review_id = path.stem
        from .imaging_history import record_import
        record_import(AppendOnlyJsonlLedger(root / 'ledger.jsonl'), args.case, review_id, result,
                      sha256_digest(args.review.read_bytes()), replaces=args.replaces_review_id,
                      reason=args.amendment_reason)
        return 0, {'path': str(path), 'review_id': review_id, 'measurement_count': len(result['measurements']),
                   'clinical_validation': 'NOT_VALIDATED'}
    if args.command == 'compare-runs':
        from .comparison import compare_results
        from .tribunal import TribunalService
        result = compare_results(TribunalService.load(root, run_id(args.before)), TribunalService.load(root, run_id(args.after)))
        return 0, {'path': str(save_json(root, 'comparisons', result)), **result}
    if args.command == 'literature':
        from .literature import LiteratureScreener, FullTextResolver, DecisionIssue
        from .evidence_appraisal import EvidenceAppraiser
        from .providers import build_live_gateways
        from .research_contracts import EvidenceCard, QuestionSet
        ledger = AppendOnlyJsonlLedger(root / 'runs' / run_id(args.run_id) / 'ledger.jsonl')
        if args.action == 'search-plan':
            plans = [r['payload'] for r in ledger.records() if r['kind'] == 'SEARCH_REVIEW_REQUIRED']
            return 0, {'run_id': args.run_id, 'plans': plans, 'instruction': 'Review for identifiers before resume --approve-search.'}
        cards = tuple(EvidenceCard.model_validate_json(json.dumps(v)) for v in read_json(args.cards))
        context = read_json(args.context) if args.context else {}
        if args.fulltext_limit < 0:
            raise ValueError('fulltext-limit must be nonnegative')
        if args.action == 'fetch':
            resolver = FullTextResolver(ledger=ledger, local_papers=read_json(args.local_papers) if args.local_papers else {})
            result, requested = [], 0
            for card in sorted(cards, key=lambda c: (-c.priority_score, c.evidence_id)):
                if card.fulltext_required and requested < args.fulltext_limit:
                    card = resolver.fetch(card, retry_unavailable=args.retry_unavailable)
                    requested += 1
                elif card.fulltext_required:
                    card = card.model_copy(update={'fulltext_status': 'PENDING_LIMIT'})
                result.append(card)
        else:
            registry = ModelRegistry.default()
            estimate = registry.estimate_deliberation_cost(20000, evidence_limit=len(cards))
            if not args.approve_cost:
                from .cli import _cost_payload
                return 2, {**_cost_payload(estimate), 'scope': 'conservative workflow estimate; actual stage calls vary'}
            estimate = approved_estimate(args, estimate)
            gateways = build_live_gateways(registry)
            if args.action == 'screen':
                screener = LiteratureScreener(gateways['anthropic'], ledger, registry)
                issues = tuple(DecisionIssue.model_validate_json(json.dumps(v)) for v in read_json(args.issues)) if args.issues else screener.issues(
                    QuestionSet.for_comprehensive_review(args.case, args.question), context, estimate)
                result = screener.screen(cards, issues, context, estimate)
                save_json(root, 'literature/issues', [i.model_dump(mode='json') for i in issues])
            else:
                appraiser = EvidenceAppraiser(gateways['anthropic'], ledger, registry)
                result = [appraiser.appraise(c, context, estimate) for c in cards if c.screening_status != 'EXCLUDE']
        path = save_json(root, 'literature/' + args.action, [c.model_dump(mode='json') for c in result])
        return 0, {'path': str(path), 'card_count': len(result)}
    return None
