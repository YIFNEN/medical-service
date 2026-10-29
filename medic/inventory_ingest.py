"""Import explicitly reviewed inventory leaves without modifying originals."""
from pathlib import Path, PurePosixPath

from .extended_cli import read_json
from .intake import FileIntakeService
from .inventory import read_record, _linked
from .ledger import AppendOnlyJsonlLedger, ContentAddressedStore, _atomic_write
from .material_roles import role_view


def ingest_selected(root, inventory, case_id, record_ids):
    root = Path(root)
    ledger = AppendOnlyJsonlLedger(root / 'ledger.jsonl')
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise ValueError('select distinct inventory record IDs')
    view = role_view(inventory, ledger)
    by_id = {r['record_id']: (index, r) for index, r in enumerate(view['records'])}
    verified = []
    # Verify all selections before staging any source or creating an intake.
    for record_id in record_ids:
        if record_id not in by_id:
            raise ValueError('unknown inventory record ID')
        index, item = by_id[record_id]
        if item['role'] != 'PATIENT_RECORD' or item['case_id'] != case_id:
            raise ValueError('explicit reviewed patient-record role for the same case required')
        record = inventory['records'][index]
        content = read_record(inventory, record)
        suffix = PurePosixPath(record['archive_path'][-1] if record['archive_path'] else record['source_path']).suffix.lower()
        if suffix not in {'.txt', '.md', '.pdf', '.png', '.jpg', '.jpeg'}:
            raise ValueError('selected record format is not supported by document intake; use dicom inspect for imaging')
        path = root / 'inventory-staging' / (record['sha256'].split(':')[1] + suffix)
        if any(p.exists() and _linked(p) for p in (path, *path.parents)):
            raise ValueError('linked staging path is not permitted')
        verified.append((record_id, item, record, content, path))
    for _, _, _, content, path in verified:
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError('staged source integrity mismatch')
        else:
            _atomic_write(path, content)
    intake = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files(case_id, [v[4] for v in verified])
    for record_id, item, record, _, _ in verified:
        ledger.append_once('INVENTORY_MATERIAL_INGESTED', case_id + ':' + record_id,
            {'case_id': case_id, 'snapshot_id': view['snapshot_id'], 'record_id': record_id,
             'record': record, 'role_event_id': item['role_event_id'], 'intake_id': intake.intake_id})
    return {'case_id': case_id, 'intake_id': intake.intake_id, 'file_count': len(intake.files),
            'block_count': len(intake.blocks), 'record_ids': list(record_ids), 'originals_modified': False,
            'paid_calls_made': 0}


def register(subparsers):
    parser = subparsers.add_parser('ingest-inventory')
    parser.add_argument('case')
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--record-id', action='append', required=True)
    parser.add_argument('--store', type=Path, default=Path('.medic'))


def execute(args):
    if args.command != 'ingest-inventory':
        return None
    return 0, ingest_selected(args.store, read_json(args.inventory), args.case, args.record_id)
