"""Append-only import lineage for already validated imaging reviews."""


def record_import(ledger, case_id, review_id, review, source_hash, *, replaces=None, reason=None):
    if (replaces is None) != (reason is None) or (reason is not None and not reason.strip()):
        raise ValueError('replacement review ID and nonempty amendment reason are required together')
    if replaces == review_id:
        raise ValueError('a review cannot replace itself')
    payload = {'case_id': case_id, 'review_id': review_id, 'review': review,
               'source_review_sha256': source_hash}
    if replaces is not None:
        payload.update(replaces_review_id=replaces, amendment_reason=reason.strip())
    key = case_id + ':' + review_id
    with ledger.key_lock('imaging-import', case_id):
        records = ledger.records()
        existing = next((r for r in records if r['kind'] == 'IMAGING_REVIEW_IMPORTED'
                         and r['natural_key'] == key), None)
        if existing:
            if existing['payload'] != payload:
                raise ValueError('existing review import has different provenance or amendment details')
            return existing['record_id']
        if replaces is not None:
            imports = [r['payload'] for r in records if r['kind'] == 'IMAGING_REVIEW_IMPORTED'
                       and r['payload'].get('case_id') == case_id]
            target = next((p for p in imports if p['review_id'] == replaces), None)
            if target is None or any(target['review'][k] != review[k] for k in ('study_uid', 'series_uid')):
                raise ValueError('replacement target must belong to the same case, study and series')
            latest = next((p for p in reversed(imports)
                           if all(p['review'][k] == review[k] for k in ('study_uid', 'series_uid'))), None)
            if latest['review_id'] != replaces:
                raise ValueError('replacement target is stale; review the latest import first')
        return ledger.append_once('IMAGING_REVIEW_IMPORTED', key, payload)
