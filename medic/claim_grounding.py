import json
import re

from .ledger import deterministic_id
from .research_contracts import ProposedSynthesis, WithheldClaim


def synthesis_claims(synthesis):
    claims = []
    def add(location, text):
        claims.append({"claim_id": deterministic_id("synthesis-claim", {"location": location, "text": text}),
                       "location": location, "text": text})
    for field in ("agreements", "disagreements", "decision_changers"):
        for index, value in enumerate(getattr(synthesis, field)):
            add(f"{field}.{index}", value)
    for index, action in enumerate(synthesis.conditional_actions):
        add(f"conditional_actions.{index}.action", action.action)
        for field in ("prerequisites", "benefits", "downsides"):
            for offset, value in enumerate(getattr(action, field)):
                add(f"conditional_actions.{index}.{field}.{offset}", value)
    return tuple(claims)


def known_patient_fact_ids(patient):
    facts = {item['fact_id'] for item in patient.get('fact_details', ()) if item.get('fact_id')}
    for review in patient.get('imaging_reviews', ()):
        if review.get('review_status') not in {'USER_REVIEWED', 'EXPERT_REVIEWED'}:
            continue
        for kind in ('measurements', 'segmentations'):
            facts.update(item['fact_id'] for item in review.get(kind, ()) if item.get('fact_id'))
    return facts


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
            if any(c.fulltext_required and c.analysis_scope != 'FULL_TEXT' for c in cited):
                reasons.append('required full text verification is incomplete')
            if audit.requires_fulltext_verification and not any(c.analysis_scope == 'FULL_TEXT' for c in cited):
                reasons.append('claim requires full text verification, not model prior knowledge or abstract alone')
            for quote in audit.source_quotes:
                card = cards.get(quote.evidence_id)
                source = (card.source_text or "\n".join(card.findings)) if card else ""
                if quote.evidence_id not in audit.evidence_ids or quote.quote not in source:
                    reasons.append("supporting quote absent from the cited source")
            if reasons:
                localized = True
                objections[audit.claim_id].extend(reasons)
        if (review.citation_errors or review.evidence_contradictions) and not localized:
            raise ValueError("evidence contradiction is not localized to a claim")
    known_urls = {str(card.url).rstrip("/") for card in evidence}
    for claim in claims:
        urls = re.findall(r"https?://[^\s<>\"\]]+", claim["text"])
        if any(url.rstrip("/.,)") not in known_urls for url in urls):
            objections[claim["claim_id"]].append("unregistered URL in claim")
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
        if path[0] == "conditional_actions":
            withheld_actions.add(path[1])
    for index in withheld_actions:
        payload["conditional_actions"][index]["action"] = "판단 보류: 행동 또는 전제·장단점의 근거 검증 미통과"
    payload["conditional_action_ranking"] = [item["action"] for item in payload["conditional_actions"]]
    return ProposedSynthesis.model_validate_json(json.dumps(payload)), tuple(withheld)
