"""Explicit, snapshot-bound material roles; no inferred patient or paper identity."""
from datetime import UTC, datetime
import re
import uuid

from .ledger import deterministic_id


def role_view(inventory, ledger):
    snapshot_id = deterministic_id('inventory', inventory)
    latest = {}
    for event in ledger.records():
        if event['kind'] == 'MATERIAL_ROLE_REVIEWED' and event['payload']['snapshot_id'] == snapshot_id:
            latest[event['payload']['record_id']] = (event['record_id'], event['payload'])
    records = []
    for record in inventory['records']:
        record_id = deterministic_id('material', {'snapshot_id': snapshot_id, 'record': record})
        event_id, review = latest.get(record_id, (None, {}))
        records.append({**record, 'record_id': record_id, 'role': review.get('role', 'UNKNOWN'),
                        'case_id': review.get('case_id'), 'citation': review.get('citation'),
                        'role_event_id': event_id, 'reviewer': review.get('reviewer'),
                        'review_reason': review.get('reason')})
    return {'snapshot_id': snapshot_id, 'records': records,
            'identity_scope': 'USER_REVIEWED_ROLE_NOT_CLINICAL_OR_BIBLIOGRAPHIC_AUTHENTICATION'}


def review_role(inventory, ledger, record_id, *, role, reviewer, reason, case_id=None, citation=None):
    if role not in {'UNKNOWN', 'PATIENT_RECORD', 'LITERATURE', 'OTHER'}:
        raise ValueError('unknown material role')
    if not reviewer.strip() or not reason.strip():
        raise ValueError('reviewer and reason must be nonempty')
    if role == 'PATIENT_RECORD' and not (case_id and case_id.strip()):
        raise ValueError('patient record requires explicit case ID')
    if role == 'LITERATURE' and not (citation and citation.strip()):
        raise ValueError('literature requires reviewed bibliographic information')
    if (role != 'PATIENT_RECORD' and case_id is not None) or (role != 'LITERATURE' and citation is not None):
        raise ValueError('case/citation does not match selected material role')
    with ledger.key_lock('material-role', record_id):
        view = role_view(inventory, ledger)
        matches = [r for r in view['records'] if r['record_id'] == record_id]
        if len(matches) != 1:
            raise ValueError('unknown or ambiguous inventory record ID')
        record = matches[0]
        if (record['status'] != 'ACCEPTED' or record['classification'] != 'UNKNOWN'
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', record.get('sha256') or '')):
            raise ValueError('only accepted hashed leaf materials can be classified')
        payload = {'snapshot_id': view['snapshot_id'], 'record_id': record_id, 'role': role,
                   'case_id': case_id.strip() if case_id else None,
                   'citation': citation.strip() if citation else None,
                   'reviewer': reviewer.strip(), 'reason': reason.strip()}
        prior = next((r for r in ledger.records() if r['record_id'] == record['role_event_id']), None)
        if prior and all(prior['payload'].get(k) == v for k, v in payload.items()):
            return prior['record_id']
        if role != 'UNKNOWN':
            from .inventory import read_record
            raw = inventory['records'][view['records'].index(record)]
            read_record(inventory, raw)
        return ledger.append_once('MATERIAL_ROLE_REVIEWED', uuid.uuid4().hex,
            {**payload, 'previous_event_id': record['role_event_id'], 'reviewed_at': datetime.now(UTC).isoformat()})
