import json
import re

from .ledger import deterministic_id
from .research_contracts import ProposedSynthesis, WithheldClaim


def synthesis_claims(synthesis):
    claims = []
    def add(location, text, identifier=None):
        if not isinstance(identifier, str) or not identifier or any(c['claim_id'] == identifier for c in claims):
            identifier = deterministic_id("synthesis-claim", {"location": location, "text": text})
        claims.append({"claim_id": identifier,
                       "location": location, "text": text})
    for field in ("agreements", "disagreements", "decision_changers"):
        for index, value in enumerate(getattr(synthesis, field)):
            add(f"{field}.{index}", value)
    for index, action in enumerate(synthesis.conditional_actions):
        add(f"conditional_actions.{index}.action", action.action)
        for field in ("prerequisites", "benefits", "downsides"):
            for offset, value in enumerate(getattr(action, field)):
                add(f"conditional_actions.{index}.{field}.{offset}", value)
    for index, claim in enumerate(synthesis.question_claims):
        add(f'question_claims.{index}.text', claim.text, claim.claim_id)
    return tuple(claims)


def known_patient_fact_ids(patient):
    facts = {item['fact_id'] for item in patient.get('fact_details', ()) if item.get('fact_id')}
    for review in patient.get('imaging_reviews', ()):
        if review.get('review_status') not in {'USER_REVIEWED', 'EXPERT_REVIEWED'}:
            continue
        for kind in ('measurements', 'segmentations'):
            facts.update(item['fact_id'] for item in review.get(kind, ()) if item.get('fact_id'))
    return facts


def _fulltext_supports(audit, cited):
    for card in cited:
        if card.analysis_scope == 'FULL_TEXT' and not audit.required_evidence_fields:
            return True
        if card.access_level != 'FULL_TEXT' or card.analysis_scope not in ('FULL_TEXT', 'PARTIAL_FULL_TEXT'):
            continue
        required = set(audit.required_evidence_fields or ('population', 'study_design', 'findings'))
        quoted = {q.quote for q in audit.source_quotes if q.evidence_id == card.evidence_id}
        verified = set()
        for field in card.field_support:
            if (field.quote in quoted and field.source_start is not None and field.source_end is not None
                    and card.source_text[field.source_start:field.source_end] == field.quote):
                verified.add(field.field)
        if required <= verified:
            return True
    return False


def ground_synthesis(synthesis, reviews, evidence, patient):
    claims = synthesis_claims(synthesis)
    expected = {item["claim_id"] for item in claims}
    cards = {card.evidence_id: card for card in evidence}
    facts = known_patient_fact_ids(patient)
    objections = {claim_id: [] for claim_id in expected}
    for review in reviews:
        ids = [item.claim_id for item in review.claim_reviews]
        if set(ids) != expected or len(ids) != len(expected):
            raise ValueError("claim review must cover every synthesis claim exactly once")
        localized = False
        for audit in review.claim_reviews:
            reasons = []
            if audit.verdict != "SUPPORTED":
                reasons.append(f"{review.reviewer_provider}: {audit.rationale}")
            if set(audit.evidence_ids) - set(cards) or set(audit.patient_fact_ids) - facts:
                reasons.append("unknown evidence or patient fact ID")
            if not audit.source_quotes and not audit.patient_fact_ids:
                reasons.append("no source quote or patient fact supporting the claim")
            cited = [cards[eid] for eid in audit.evidence_ids if eid in cards]
            needs_fulltext = audit.requires_fulltext_verification or bool(audit.required_evidence_fields) or any(c.fulltext_required for c in cited)
            if needs_fulltext and not _fulltext_supports(audit, cited):
                reasons.append('claim requires full text verification of necessary methods/results; verified passages are incomplete')
            if any(c.publication_status == 'RETRACTED' or (c.publication_status in {'CORRECTION_REQUIRES_REVIEW', 'EXPRESSION_OF_CONCERN'}
                    and not c.publication_notice_reviewed) for c in cited):
                reasons.append('publication notice unresolved or source retracted')
            for quote in audit.source_quotes:
                card = cards.get(quote.evidence_id)
                source = (card.source_text or "\n".join(card.findings)) if card else ""
                if quote.evidence_id not in audit.evidence_ids or quote.quote not in source:
                    reasons.append("supporting quote absent from the cited source")
                elif card.analysis_scope == 'PARTIAL_FULL_TEXT' and not any(
                        quote.quote in field.quote for field in card.field_support):
                    reasons.append('supporting quote was not in an analyzed full text passage')
            if reasons:
                localized = True
                objections[audit.claim_id].extend(reasons)
        if (review.citation_errors or review.evidence_contradictions) and not localized:
            raise ValueError("evidence contradiction is not localized to a claim")
    known_urls = {str(card.url).rstrip("/") for card in evidence}
    questions = patient.get('scoped_questions', patient.get('typed_questions', ()))
    if questions:
        from .question_runtime import available_references
        from .questions import validate_question_claims
        results = validate_question_claims(questions, synthesis.question_claims,
            available=available_references(patient, evidence))
        for index, result in enumerate(results):
            claim = next(c for c in claims if c['location'] == f'question_claims.{index}.text')
            if not result.accepted:
                objections[claim['claim_id']].extend(result.reasons)
        known_questions = {q['question_id'] for q in questions}
        question_claim_ids = {c.claim_id for c in synthesis.question_claims}
        for index, action in enumerate(synthesis.conditional_actions):
            claim = next(c for c in claims if c['location'] == f'conditional_actions.{index}.action')
            if not action.question_ids or set(action.question_ids) - known_questions:
                objections[claim['claim_id']].append('action is not linked to allowed questions')
            if not action.claim_ids or set(action.claim_ids) - question_claim_ids:
                objections[claim['claim_id']].append('action is not linked to supporting question claims')
            if any(objections.get(identifier) for identifier in action.claim_ids):
                objections[claim['claim_id']].append('action depends on a withheld question claim')
    for claim in claims:
        from .questions import _NUMERIC_TEXT
        if not claim['location'].startswith('question_claims.') and _NUMERIC_TEXT.search(claim['text']):
            objections[claim['claim_id']].append('personal numeric estimate requires a typed question and validated numeric support')
        urls = re.findall(r"https?://[^\s<>\"\]]+", claim["text"])
        if any(url.rstrip("/.,)") not in known_urls for url in urls):
            objections[claim["claim_id"]].append("unregistered URL in claim")
    if questions:
        by_id = {c.claim_id: c for c in synthesis.question_claims}
        for index, action in enumerate(synthesis.conditional_actions):
            target = next(c for c in claims if c['location'] == f'conditional_actions.{index}.action')
            for identifier in action.claim_ids:
                dependency = by_id.get(identifier)
                if (dependency is None or dependency.status not in ('ANSWER', 'CONDITIONAL')
                        or dependency.question_id not in action.question_ids or objections.get(identifier)):
                    objections[target['claim_id']].append('action depends on a non-answer, mismatched or rejected claim')
    payload = synthesis.model_dump(mode="json")
    withheld = []
    withheld_actions = set()
    for claim in claims:
        reasons = objections[claim["claim_id"]]
        if not reasons:
            continue
        withheld.append(WithheldClaim(claim_id=claim["claim_id"], location=claim["location"],
                                     original_text=claim["text"], reasons=tuple(dict.fromkeys(reasons))))
        path = [int(item) if item.isdigit() else item for item in claim["location"].split(".")]
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = f"판단 보류 ({claim['claim_id']})"
        if path[0] == 'question_claims':
            target['status'] = 'MISSING_DATA'
        if path[0] == "conditional_actions":
            withheld_actions.add(path[1])
    for index in withheld_actions:
        payload["conditional_actions"][index]["action"] = "판단 보류: 행동 또는 전제·장단점의 근거 검증 미통과"
    payload["conditional_action_ranking"] = [item["action"] for item in payload["conditional_actions"]]
    return ProposedSynthesis.model_validate_json(json.dumps(payload)), tuple(withheld)
