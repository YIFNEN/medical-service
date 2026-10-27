"""Descriptive run comparison; never infer clinical improvement from a diff."""
from .ledger import canonical_json_bytes, sha256_digest


def compare_results(before, after):
    if before.case_id != after.case_id:
        raise ValueError('cannot compare different cases')
    def identity(item):
        return sha256_digest(canonical_json_bytes(item))
    def changes(old, new):
        a = {identity(item): item for item in old}
        b = {identity(item): item for item in new}
        return {'added': [b[k] for k in b.keys() - a.keys()], 'removed': [a[k] for k in a.keys() - b.keys()]}
    def evidence(result):
        return [{'evidence_id': c.evidence_id, 'source_sha256': c.source_sha256,
                 'analysis_scope': c.analysis_scope, 'fulltext_status': c.fulltext_status,
                 'source_text_hash': identity(c.source_text)} for c in result.evidence_cards]
    images_before = before.patient_context.get('imaging_reviews', [])
    images_after = after.patient_context.get('imaging_reviews', [])
    def duration(result):
        return (result.created_at - result.data_cutoff).total_seconds() if result.created_at and result.data_cutoff else None
    return {
        'version': 'run-comparison/1', 'case_id': before.case_id, 'before': before.run_id, 'after': after.run_id,
        'facts': changes(before.patient_context.get('fact_details', []), after.patient_context.get('fact_details', [])),
        'evidence': changes(evidence(before), evidence(after)),
        'imaging_changed': images_before != images_after, 'imaging': changes(images_before, images_after),
        'opinions_changed': before.synthesis.model_dump(mode='json') != after.synthesis.model_dump(mode='json'),
        'clinical_change': 'NOT_INFERRED',
        'warning': 'Acquisition, review and evidence may differ; changed output does not prove clinical progression or improved accuracy.',
        'estimated_usage_usd': {'before': sum(c.estimated_cost_usd for c in before.calls), 'after': sum(c.estimated_cost_usd for c in after.calls)},
        'elapsed_seconds_including_resume_delays': {'before': duration(before), 'after': duration(after)},
        'accuracy_assessment': 'Requires independently reviewed claim support, numerical accuracy and applicability labels.',
    }
