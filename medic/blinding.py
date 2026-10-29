"""Clinician-withheld inputs, immutable result seals, and post-seal comparison inputs."""
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .extended_cli import save_json, run_id as validate_run_id
from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, sha256_digest
from .research_contracts import StrictFrozenModel


class BlindApproval(StrictFrozenModel):
    schema_version: Literal['medic-blind-approval/1']
    case_id: str
    reviewer: str
    reason: str
    questions_reviewed: bool
    profile_items: tuple[str, ...] = ()
    imaging_review_ids: tuple[str, ...] = ()


def _ledger(root, run_id):
    return AppendOnlyJsonlLedger(Path(root) / 'runs' / validate_run_id(run_id) / 'ledger.jsonl')


def _snapshot(root, run_id):
    snapshot = next((r['payload'] for r in _ledger(root, run_id).records() if r['kind'] == 'BLIND_INPUT_CREATED'), None)
    if snapshot is not None:
        if snapshot['run_id'] != run_id:
            raise ValueError('blind snapshot belongs to another run')
        unsigned = {k: v for k,v in snapshot.items() if k != 'snapshot_sha256'}
        if sha256_digest(canonical_json_bytes(unsigned)) != snapshot['snapshot_sha256']:
            raise ValueError('blind snapshot integrity mismatch')
    return snapshot


def _project_measurement(item):
    allowed = ('fact_id', 'measurement_id', 'kind', 'value', 'units', 'coordinate_system', 'points_lps',
               'plane', 'protocol', 'axes', 'confirmation_scope', 'metadata_status', 'source_conditions')
    result = {key: item[key] for key in allowed if key in item}
    for key, fields in {'axes': ('x', 'y', 'source'), 'plane': ('origin', 'normal'),
                        'source_conditions': ('modality', 'laterality', 'weight_bearing')}.items():
        value = item.get(key)
        if isinstance(value, dict):
            result[key] = {field: value[field] for field in fields if field in value}
    metadata = item.get('metadata', {})
    result['metadata'] = {key: metadata[key] for key in ('anatomy', 'laterality', 'metric', 'method',
                          'protocol_version', 'plane_definition') if isinstance(metadata.get(key), str)}
    if isinstance(metadata.get('landmarks'), list) and all(isinstance(v, str) for v in metadata['landmarks']):
        result['metadata']['landmarks'] = metadata['landmarks']
    acquisition = metadata.get('acquisition') or {}
    result['metadata']['acquisition'] = {key: acquisition[key]
        for key in ('modality', 'weight_bearing', 'position', 'protocol') if isinstance(acquisition.get(key), str)}
    return result


def _withheld(root, snapshot):
    # The content-addressed basename is pinned; a caller cannot redirect it to another case.
    path = Path(root) / 'withheld-inputs' / (snapshot['withheld_sha256'].split(':')[1] + '.json')
    content = path.read_bytes()
    if sha256_digest(content) != snapshot['withheld_sha256']:
        raise ValueError('withheld input integrity mismatch')
    value = json.loads(content)
    if value['case_id'] != snapshot['case_id'] or value['run_id'] != snapshot['run_id']:
        raise ValueError('withheld input belongs to another case or run')
    return value


def create_blind_input(root, run_id, case_id, context, questions, approval):
    approval = BlindApproval.model_validate_json(json.dumps(approval))
    if approval.case_id != case_id or not approval.questions_reviewed or not approval.reviewer.strip() or not approval.reason.strip():
        raise ValueError('explicit same-case question/visibility review required')
    case = context.get('case_context', {})
    if case.get('case_id') != case_id:
        raise ValueError('case context binding required')
    reviews = {r['review']['fact_id']: r['review'] for r in case.get('review_inputs', ())}
    allowed_types = {'observation', 'procedure', 'self_report', 'actual_exposure', 'derived_measure'}
    facts = {f.get('fact_id'): f for f in context.get('fact_details', ()) if f.get('fact_id')}
    eligibility = {}
    def eligible(identifier, visiting=frozenset()):
        if identifier in eligibility:
            return eligibility[identifier]
        if identifier in visiting or identifier not in facts:
            return False
        fact, review = facts[identifier], reviews.get(identifier, {})
        ok = (fact.get('user_confirmed') is True and review.get('classification') in allowed_types
              and review.get('context', {}).get('case_id') == case_id
              and fact.get('scope', {}).get('source_kind') != 'CLINICIAN_OPINION')
        parents = list(review.get('derived_from_ids', ()))
        if (review.get('duplicate') or {}).get('relation') in ('copied', 'corrected'):
            parents.append(review['duplicate']['fact_id'])
        eligibility[identifier] = ok and all(eligible(p, visiting | {identifier}) for p in parents)
        return eligibility[identifier]
    allowed, hidden = [], []
    for fact in context.get('fact_details', ()):
        review = reviews.get(fact.get('fact_id'), {})
        permitted = eligible(fact.get('fact_id'))
        if permitted:
            allowed.append(fact)
        else:
            hidden.append({'fact': fact, 'classification': review.get('classification', 'UNCLASSIFIED')})
    eligible_ids = {f.get('fact_id') for f in allowed}
    profile = case.get('profile')
    visible_profile = {}
    if profile:
        if case.get('profile_stale'):
            raise ValueError('profile references stale reviewed facts; re-review before blind use')
        for dimension in ('symptoms', 'function', 'activity', 'constraints', 'goals', 'priorities'):
            for item in profile['profile'].get(dimension, ()):
                if dimension + ':' + item['name'] in approval.profile_items:
                    if set(item.get('fact_ids', ())) - eligible_ids:
                        raise ValueError('profile refers to withheld facts')
                    visible_profile.setdefault(dimension, []).append(item)
    imaging = []
    for review in context.get('imaging_reviews', ()):
        if review.get('review_id') in approval.imaging_review_ids:
            projected = {key: review[key] for key in ('review_id', 'review_status') if key in review}
            projected['measurements'] = [_project_measurement(m) for m in review.get('measurements', ())]
            projected['conditions'] = {key: review.get('conditions', {})[key] for key in
                ('modality', 'weight_bearing', 'laterality', 'dates', 'geometry') if key in review.get('conditions', {})}
            if isinstance(projected['conditions'].get('geometry'), dict):
                geometry = projected['conditions']['geometry']
                projected['conditions']['geometry'] = {key: geometry[key] for key in
                    ('affine_lps', 'shape', 'slice_order', 'image_plane_only') if key in geometry}
            projected['segmentations'] = [{key: s[key] for key in ('fact_id', 'sha256', 'label', 'metrics') if key in s}
                                           for s in review.get('segmentations', ())]
            imaging.append(projected)
    visible = {'confirmed_facts': [f"{f['fact_key']}: {f['value']}" for f in allowed],
               'fact_details': allowed, 'uncertainties': ['Unclassified or clinician-origin assertions withheld; absence is not negative evidence.'],
               'imaging_reviews': imaging,
               'case_context': {'case_id': case_id, 'profile': visible_profile,
                                'review_inputs': [{'event_id': r.get('event_id'), 'version': r.get('version'),
                                                  'review': r['review']} for r in case.get('review_inputs', ())
                                                  if r['review']['fact_id'] in eligible_ids]}}
    for key in ('typed_questions', 'question_evaluations', 'question_version'):
        if key in context:
            visible[key] = context[key]
    if visible.get('typed_questions'):
        from .questions import evaluate_questions
        from .question_runtime import available_references
        visible['question_evaluations'] = [r.model_dump(mode='json') for r in evaluate_questions(
            visible['typed_questions'], available=available_references(visible))]
    hidden_bundle = {'version': 'medic-withheld-input/1', 'case_id': case_id, 'run_id': run_id,
                     'facts': hidden, 'source_context': context, 'approval': approval.model_dump(mode='json')}
    _check_hidden(hidden_bundle, json.dumps({'context': visible, 'questions': questions}, ensure_ascii=False))
    hidden_hash = sha256_digest(canonical_json_bytes(hidden_bundle))
    ledger = _ledger(root, run_id)
    existing = _snapshot(root, run_id)
    snapshot = {'version': 'medic-blind-input/1', 'case_id': case_id, 'run_id': run_id,
                'mode': 'CLINICIAN_BLIND', 'visible_context': visible, 'questions': questions,
                'withheld_sha256': hidden_hash,
                'input_cutoff': existing['input_cutoff'] if existing else datetime.now(UTC).isoformat()}
    snapshot['snapshot_sha256'] = sha256_digest(canonical_json_bytes(snapshot))
    with ledger.key_lock('blind-input', run_id):
        if existing and existing != snapshot:
            raise ValueError('blind input is pinned; create a fresh run')
        save_json(root, 'withheld-inputs', hidden_bundle)
        ledger.append_once('BLIND_INPUT_CREATED', run_id, snapshot)
    visible['analysis_input'] = {k: snapshot[k] for k in ('mode', 'snapshot_sha256', 'input_cutoff')}
    assert_no_leak(root, run_id, json.dumps({'context': visible, 'questions': questions}, ensure_ascii=False))
    return visible


def _check_hidden(hidden, text):
    json_escapes = re.compile(r'(?:\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))+')
    def normalize(s):
        # Prompts may contain JSON embedded in string-valued metadata. Decode
        # valid JSON escapes through each layer before exact-phrase matching.
        # Cap repeated scans at 16 layers; deeper encoding fails closed rather
        # than allowing an untrusted backslash chain to consume unbounded work.
        for _ in range(16):
            decoded = json_escapes.sub(lambda m: json.loads('"' + m.group(0) + '"'), s)
            if decoded == s:
                break
            s = decoded
        else:
            if json_escapes.search(s):
                raise ValueError('blind input exceeds 16 JSON escape decoding layers')
        return ' '.join(unicodedata.normalize('NFC', s).casefold().split())
    candidate = normalize(text)
    for item in hidden['facts']:
        phrase = normalize(str(item['fact'].get('value', '')))
        if len(phrase) >= 8 and phrase in candidate:
            raise ValueError('withheld content detected; review input separation locally')


def assert_no_leak(root, run_id, text):
    snapshot = _snapshot(root, run_id)
    if snapshot is not None:
        _check_hidden(_withheld(root, snapshot), text)


def seal_result(root, result):
    snapshot = _snapshot(root, result.run_id)
    if snapshot is None or result.patient_context.get('analysis_input', {}).get('mode') != 'CLINICIAN_BLIND':
        raise ValueError('result has no clinician-blind input snapshot')
    visible = dict(snapshot['visible_context'])
    visible['analysis_input'] = {k: snapshot[k] for k in ('mode', 'snapshot_sha256', 'input_cutoff')}
    # Workflow may append literature traces but must not alter the reviewed patient view.
    if (result.case_id != snapshot['case_id'] or any(result.patient_context.get(k) != v for k,v in visible.items())
            or result.question_set.model_dump(mode='json') != snapshot['questions']):
        raise ValueError('result input differs from pinned blind snapshot')
    if ({o.provider for o in result.opinions} != {'openai', 'anthropic', 'google'}
            or {r.reviewer_provider for r in result.synthesis_reviews} != {'openai', 'anthropic', 'google'}
            or len(result.synthesis_reviews) != 3):
        raise ValueError('incomplete provider review cannot be sealed')
    payload = result.model_dump(mode='json')
    completed = next((r['payload']['result'] for r in _ledger(root, result.run_id).records()
                      if r['kind'] == 'TRIBUNAL_COMPLETED'), None)
    if completed != payload:
        raise ValueError('only the persisted completed result can be sealed')
    assert_no_leak(root, result.run_id, json.dumps(payload))
    seal = {'version': 'medic-blind-seal/1', 'case_id': result.case_id, 'run_id': result.run_id,
            'snapshot_sha256': snapshot['snapshot_sha256'], 'result_sha256': sha256_digest(canonical_json_bytes(payload)),
            'input_cutoff': snapshot['input_cutoff'], 'completed_at': result.created_at.isoformat()}
    _ledger(root, result.run_id).append_once('BLIND_RESULT_SEALED', result.run_id, seal)
    return seal


def comparison_input(root, run_id):
    ledger = _ledger(root, run_id)
    seal = next((r['payload'] for r in ledger.records() if r['kind'] == 'BLIND_RESULT_SEALED'), None)
    if seal is None:
        raise ValueError('a sealed clinician-blind result is required before comparison')
    if seal['run_id'] != run_id:
        raise ValueError('sealed result belongs to another requested run')
    from .tribunal import TribunalService
    from .claim_grounding import synthesis_claims
    result = TribunalService.load(root, run_id)
    if result.run_id != run_id or result.case_id != seal['case_id']:
        raise ValueError('result belongs to another case or run')
    if seal_result(root, result) != seal:
        raise ValueError('sealed result integrity mismatch')
    hidden = _withheld(root, _snapshot(root, run_id))
    return {'case_id': result.case_id, 'run_id': run_id, 'seal': seal,
            'claims': list(synthesis_claims(result.original_synthesis or result.synthesis)),
            'withheld_claim_ids': [claim.claim_id for claim in result.withheld_claims],
            'clinician_facts': [i['fact'] for i in hidden['facts'] if i['classification'] == 'clinician_assertion'],
            'comparison_scope': 'OPINION_COMPARISON_NOT_CLINICAL_ACCURACY', 'agreement_accuracy_score': None}
