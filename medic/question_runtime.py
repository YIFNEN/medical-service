"""Freeze per-run question scopes and validate generated claims independently."""
from .questions import AvailableReferences, CaseQuestionService, evaluate_questions, validate_question_claims
from .ledger import deterministic_id


def available_references(context, cards=()):
    facts = frozenset(f['fact_id'] for f in context.get('fact_details', ()) if f.get('fact_id') and f.get('user_confirmed'))
    measurements = frozenset(m['fact_id'] for review in context.get('imaging_reviews', ())
        if review.get('review_status') in ('USER_REVIEWED', 'EXPERT_REVIEWED')
        for m in review.get('measurements', ()) if m.get('fact_id'))
    blocked = frozenset(context.get('case_context', {}).get('ineligible_fact_ids', ()))
    return AvailableReferences(fact_ids=facts - blocked, measurement_ids=measurements - blocked,
                               evidence_ids=frozenset(c.evidence_id for c in cards))


def attach_questions(ledger, case_id, context):
    projection = CaseQuestionService(ledger).load(case_id, available=available_references(context))
    if not projection.active:
        return context
    return {**context, 'typed_questions': [q.model_dump(mode='json') for q in projection.active.bundle.questions],
            'question_evaluations': [q.model_dump(mode='json') for q in projection.evaluations],
            'question_version': {'event_id': projection.active.event_id, 'version': projection.version}}


def scope_questions(context, cards, issues):
    scoped = []
    for q in context.get('typed_questions', ()):
        identifiers = {i.issue_id for i in issues if q['question_id'] in getattr(i, 'question_ids', ())}
        eligible = [c.evidence_id for c in cards if identifiers & set(c.decision_issue_ids)]
        explicit = q.get('allowed_evidence_ids')
        scoped.append({**q, 'allowed_evidence_ids': [eid for eid in explicit if eid in eligible] if explicit else eligible})
    return scoped


def question_claim_trace(context, synthesis, cards, withheld=()):
    questions = context.get('scoped_questions', context.get('typed_questions', ()))
    if not questions:
        return {}
    results = validate_question_claims(questions, synthesis.question_claims, available=available_references(context, cards))
    withheld_by_id = {c.claim_id: c for c in withheld}
    presented = []
    for claim, result in zip(synthesis.question_claims, results):
        claim = claim.model_dump(mode='json')
        if result is not None:
            blocked = withheld_by_id.get(result.claim_id)
            presented.append({'claim_id': result.claim_id, 'question_id': result.question_id,
                'status': result.status if not blocked else 'CONFLICT', 'accepted': result.accepted and not blocked,
                'text': claim['text'] if result.accepted and not blocked else '판단 보류: 질문별 사용 자격 또는 근거 검증 미충족',
                'reasons': list(result.reasons) + (list(blocked.reasons) if blocked else []), 'citations': claim.get('citations', {}),
                'uncertainty': claim.get('uncertainty', {})})
    covered = {c.question_id for c in synthesis.question_claims}
    for q in questions:
        if q['question_id'] not in covered:
            presented.append({'claim_id': deterministic_id('missing-question-claim', q),
                'question_id': q['question_id'], 'status': 'SYSTEM_FAILURE', 'accepted': False,
                'text': '제공자가 이 질문의 구조화 답변을 반환하지 않음', 'reasons': ['missing_question_response']})
    return {'version': 'medic-question-trace/1', 'question_version': context.get('question_version'),
            'questions': questions, 'evaluations': [r.model_dump(mode='json') for r in
                evaluate_questions(questions, available=available_references(context, cards))],
            'claims': presented, 'eligibility_scope': 'structural eligibility; clinical support is reviewed separately'}
