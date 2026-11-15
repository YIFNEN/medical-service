"""Append-only imaging source assignments and review lineage."""


def bind_series(ledger, case_id, source, *, reviewer, reason, manifest_hash):
    if not case_id.strip() or not reviewer.strip() or not reason.strip():
        raise ValueError('nonempty case, reviewer and case-link review reason required')
    identity = _source_identity(source, manifest_hash)
    key = f"{identity['study_uid']}:{identity['series_uid']}"
    hashes = set(identity['original_hashes'])
    with ledger.key_lock('imaging-series-binding', 'store'):
        records = ledger.records()
        for record in records:
            payload = record['payload']
            if record['kind'] == 'IMAGING_SERIES_BOUND':
                registered = payload['source']
            elif record['kind'] == 'IMAGING_REVIEW_IMPORTED':
                registered = payload['review']
            else:
                continue
            if payload['case_id'] != case_id and (registered['study_uid'] == source['study_uid']
                    or hashes.intersection(registered.get('original_hashes', ()))):
                raise ValueError('imaging source is already associated with another case')
        existing = next((r for r in records if r['kind'] == 'IMAGING_SERIES_BOUND'
                         and r['natural_key'] == key), None)
        if existing:
            if existing['payload']['source'] != identity:
                raise ValueError('reviewed source binding has different original hashes or inspection manifest')
            return existing['record_id']
        return ledger.append_once('IMAGING_SERIES_BOUND', key, {
            'case_id': case_id, 'source': identity, 'reviewer': reviewer.strip(), 'reason': reason.strip(),
        })


def _source_identity(source, manifest_hash):
    return {'study_uid': source['study_uid'], 'series_uid': source['series_uid'],
            'original_hashes': source['original_hashes'], 'manifest_sha256': manifest_hash}


def source_binding(ledger, case_id, review, manifest_hash):
    identity = _source_identity(review, manifest_hash)
    match = next((r for r in ledger.records() if r['kind'] == 'IMAGING_SERIES_BOUND'
                  and r['payload']['case_id'] == case_id and r['payload']['source'] == identity), None)
    if match is None:
        raise ValueError('reviewed source binding for the same case and original hashes is required')
    return match['record_id']


def record_import(ledger, case_id, review_id, review, source_hash, *, replaces=None, reason=None, binding_id=None):
    if (replaces is None) != (reason is None) or (reason is not None and not reason.strip()):
        raise ValueError('replacement review ID and nonempty amendment reason are required together')
    if replaces == review_id:
        raise ValueError('a review cannot replace itself')
    payload = {'case_id': case_id, 'review_id': review_id, 'review': review,
               'source_review_sha256': source_hash}
    if binding_id is not None:
        payload['source_binding_id'] = binding_id
    if replaces is not None:
        payload.update(replaces_review_id=replaces, amendment_reason=reason.strip())
    key = case_id + ':' + review_id
    with ledger.key_lock('imaging-import', case_id):
        records = ledger.records()
        existing = next((r for r in records if r['kind'] == 'IMAGING_REVIEW_IMPORTED'
                         and r['natural_key'] == key), None)
        if existing:
            previous = dict(existing['payload'])
            if binding_id is not None:
                previous.setdefault('source_binding_id', binding_id)
            if previous != payload:
                raise ValueError('existing review import has different provenance or amendment details')
            return existing['record_id']
        imports = [r['payload'] for r in records if r['kind'] == 'IMAGING_REVIEW_IMPORTED'
                   and r['payload'].get('case_id') == case_id]
        latest = next((p for p in reversed(imports)
                       if all(p['review'][k] == review[k] for k in ('study_uid', 'series_uid'))), None)
        if latest is not None and replaces is None:
            raise ValueError('a new review of this series requires an explicit replacement and reason')
        if replaces is not None:
            target = next((p for p in imports if p['review_id'] == replaces), None)
            if target is None or any(target['review'][k] != review[k] for k in ('study_uid', 'series_uid')):
                raise ValueError('replacement target must belong to the same case, study and series')
            if latest['review_id'] != replaces:
                raise ValueError('replacement target is stale; review the latest import first')
        return ledger.append_once('IMAGING_REVIEW_IMPORTED', key, payload)
