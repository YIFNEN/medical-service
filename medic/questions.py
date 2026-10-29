"""Bounded offline M3 contracts, deterministic gates, and immutable question history.

Question IDs are caller supplied and never derived from text. AvailableReferences
is a trusted caller input, independent of a question's allowed IDs. No facts,
preferences, clinical answers, probabilities, or causal estimates are generated.

Public API:
    evaluate_question(question, *, available, decision_mode="SINGLE_BEST")
    evaluate_questions(questions, *, available)
    validate_question_claims(questions, claims, *, available)
    CaseQuestionService(ledger).save(bundle, *, available) -> QuestionEvent
    service.load(case_id, *, available=None) -> QuestionProjection
    service.history(case_id) -> tuple[QuestionEvent, ...]
    service.withdraw(case_id, event_id, *, reviewer, reason) -> QuestionEvent

All contract inputs accept models or JSON-shaped mappings. Numeric support is an
explicit reviewed attestation, not independent verification of model validity.
Numeric text detection is deliberately bounded; callers must type quantitative
claims. No unstructured text parser can establish clinical meaning or validity.

Parent CLI integration (no import of medic.cli at module import time):
    register_questions(case_subparsers)
    execute_questions(args) -> (exit_code, JSON-safe summary) | None
    case questions --case CASE [--file BUNDLE.json] [--store .medic]

BUNDLE.json is QuestionBundle: case_id, reviewer, reason, questions, optional
claims/additional_tests/replaces. Revision requires the current event ID.
Withdrawal uses operation="withdraw", replaces=EVENT_ID, and empty content.
Status excludes free text and audit values. Execution alone lazily reads
medic.cli._latest_context; the bundle cannot whitelist its own references.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Iterable, Literal, TypeVar

from pydantic import ConfigDict, Field, JsonValue, StringConstraints, field_serializer, model_validator

from .contracts import deep_freeze
from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, deterministic_id
from .model_base import StrictFrozenModel


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# References retain their exact bytes; whitespace around an ID is invalid.
Identifier = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:[\s\S]*\S)?$")]
QuestionType = Literal["CURRENT_STATE", "PROGNOSIS", "TREATMENT_EFFECT", "DECISION"]
ClaimStatus = Literal["ANSWER", "CONDITIONAL", "MISSING_DATA", "OUT_OF_SCOPE", "CONFLICT", "SYSTEM_FAILURE"]
DecisionMode = Literal["SINGLE_BEST", "OPTIONS"]
NumericKind = Literal["INDIVIDUAL_RISK", "NNT", "ABSOLUTE_CAUSAL_EFFECT"]
LEDGER_KIND = "CASE_QUESTIONS_VERSIONED"


class AvailableReferences(StrictFrozenModel):
    """Independent, frozen namespaces; membership never implies clinical support."""
    fact_ids: frozenset[Identifier] = frozenset()
    measurement_ids: frozenset[Identifier] = frozenset()
    evidence_ids: frozenset[Identifier] = frozenset()

    @field_serializer("fact_ids", "measurement_ids", "evidence_ids", when_used="json")
    def sorted_ids(self, value):
        return sorted(value)


class CitationReferences(StrictFrozenModel):
    fact_ids: tuple[Identifier, ...] = ()
    measurement_ids: tuple[Identifier, ...] = ()
    evidence_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_references(self):
        for field in ("fact_ids", "measurement_ids", "evidence_ids"):
            if len(getattr(self, field)) != len(set(getattr(self, field))):
                raise ValueError("citation references must be unique within each namespace")
        return self


class QuestionScenario(StrictFrozenModel):
    target: Text | None = None
    assessment_time: Text | None = None
    horizon: Text | None = None
    outcome: Text | None = None
    treatment_state: Text | None = None
    intervention: Text | None = None
    comparator: Text | None = None
    time_zero: Text | None = None


class QuestionContract(QuestionScenario):
    schema_version: Literal["question/1"] = "question/1"
    question_id: Identifier
    type: QuestionType
    text: Text
    options: tuple[Text, ...] = ()
    goals: tuple[Text, ...] = ()
    # None means unprovided; () explicitly records no constraints.
    constraints: tuple[Text, ...] | None = None
    priorities: tuple[Text, ...] = ()
    allowed_fact_ids: tuple[Identifier, ...] = ()
    allowed_measurement_ids: tuple[Identifier, ...] = ()
    allowed_evidence_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_allowed_ids(self):
        for name in ("allowed_fact_ids", "allowed_measurement_ids", "allowed_evidence_ids", "options"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError("allowed references and options must be unique")
        return self


class UncertaintyAxes(StrictFrozenModel):
    data: Text = "NOT_ASSESSED"
    measurement: Text = "NOT_ASSESSED"
    model: Text = "NOT_ASSESSED"
    literature: Text = "NOT_ASSESSED"
    applicability: Text = "NOT_ASSESSED"


class NumericEstimate(StrictFrozenModel):
    """Risk/effect values are proportions; NNT is a count. Never computed here."""
    kind: NumericKind
    value: float

    @model_validator(mode="after")
    def valid_bounds(self):
        if self.kind == "INDIVIDUAL_RISK" and not 0 <= self.value <= 1:
            raise ValueError("individual risk must be a probability between zero and one")
        if self.kind == "NNT" and self.value < 1:
            raise ValueError("NNT must be at least one")
        if self.kind == "ABSOLUTE_CAUSAL_EFFECT" and not -1 <= self.value <= 1:
            raise ValueError("absolute causal effect must be a probability difference")
        return self


class NumericSupport(StrictFrozenModel):
    model_config = ConfigDict(protected_namespaces=("model_dump",))
    validated_model_id: Identifier | None = None
    model_validated: bool = False
    validation_source_id: Identifier | None = None
    calibration_source_id: Identifier | None = None
    method_source_id: Identifier | None = None
    scenario: QuestionScenario = Field(default_factory=QuestionScenario)


class QuestionClaim(StrictFrozenModel):
    schema_version: Literal["question-claim/1"] = "question-claim/1"
    claim_id: Identifier
    question_id: Identifier
    text: Text
    status: ClaimStatus
    citations: CitationReferences = Field(default_factory=CitationReferences)
    uncertainty: UncertaintyAxes = Field(default_factory=UncertaintyAxes)
    numeric_estimate: NumericEstimate | None = None
    numeric_support: NumericSupport | None = None
    decision_mode: DecisionMode = "SINGLE_BEST"


class TestResultChange(StrictFrozenModel):
    result: Text
    change: Text


class AdditionalTestProposal(StrictFrozenModel):
    schema_version: Literal["additional-test-proposal/1"] = "additional-test-proposal/1"
    proposal_id: Identifier
    question_id: Identifier
    test: Text
    question: Text
    results_change: tuple[TestResultChange, ...] = Field(min_length=1)
    burdens: tuple[Text, ...] = Field(min_length=1)
    urgency: Text
    alternatives: tuple[Text, ...] = Field(min_length=1)
    citations: CitationReferences = Field(default_factory=CitationReferences)


class QuestionEvaluation(StrictFrozenModel):
    schema_version: Literal["question-evaluation/1"] = "question-evaluation/1"
    question_id: Identifier
    status: ClaimStatus
    missing_fields: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class ClaimValidation(StrictFrozenModel):
    schema_version: Literal["question-claim-validation/1"] = "question-claim-validation/1"
    claim_id: Identifier | None
    question_id: Identifier | None
    accepted: bool
    status: ClaimStatus
    reasons: tuple[str, ...] = ()


Model = TypeVar("Model", bound=StrictFrozenModel)


def _parse(model: type[Model], value) -> Model:
    # Revalidate even prebuilt/model_construct/model_copy inputs. JSON mode
    # accepts array containers for tuples without coercing strings to numbers.
    if isinstance(value, StrictFrozenModel):
        value = value.model_dump(mode="json", warnings="error")
    return model.model_validate_json(canonical_json_bytes(_json_value(value)))


def _json_value(value):
    if isinstance(value, StrictFrozenModel):
        return value.model_dump(mode="json", warnings="error")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def _reference_reasons(citations, question, available):
    reasons = []
    for kind in ("fact", "measurement", "evidence"):
        references = getattr(citations, f"{kind}_ids")
        known = getattr(available, f"{kind}_ids")
        allowed = getattr(question, f"allowed_{kind}_ids") if question else references
        for reference in sorted(references):
            if reference not in known:
                reasons.append(f"unknown_{kind}_reference:{reference}")
            elif reference not in allowed:
                reasons.append(f"disallowed_{kind}_reference:{reference}")
    return tuple(reasons)


def _allowed(question):
    return CitationReferences(fact_ids=question.allowed_fact_ids,
        measurement_ids=question.allowed_measurement_ids, evidence_ids=question.allowed_evidence_ids)


_REQUIRED = {
    "CURRENT_STATE": ("target", "assessment_time"),
    "PROGNOSIS": ("target", "assessment_time", "horizon", "outcome", "treatment_state"),
    "TREATMENT_EFFECT": ("target", "assessment_time", "horizon", "outcome", "intervention", "comparator", "time_zero"),
    "DECISION": ("target", "assessment_time", "options", "goals", "constraints", "priorities"),
}


def evaluate_question(question: QuestionContract | dict, *, available: AvailableReferences,
                      decision_mode: DecisionMode = "SINGLE_BEST") -> QuestionEvaluation:
    """ANSWER denotes structural readiness, never a generated clinical answer."""
    question = _parse(QuestionContract, question)
    available = _parse(AvailableReferences, available)
    if decision_mode not in ("SINGLE_BEST", "OPTIONS"):
        raise ValueError("unknown decision mode")
    required = _REQUIRED[question.type]
    if question.type == "DECISION" and decision_mode == "OPTIONS":
        required = ("target", "assessment_time", "options")
    missing = tuple(name for name in required if (
        getattr(question, name) is None if name == "constraints" else
        len(question.options) < 2 if name == "options" else not getattr(question, name)))
    if question.type == 'CURRENT_STATE' and not question.allowed_fact_ids and not question.allowed_measurement_ids:
        missing += ('observations_or_measurements',)
    reasons = _reference_reasons(_allowed(question), None, available)
    return QuestionEvaluation(question_id=question.question_id,
        status="CONFLICT" if reasons else "MISSING_DATA" if missing else "ANSWER",
        missing_fields=missing, reasons=(*reasons, *(f"missing:{name}" for name in missing)))


def _questions_by_id(questions):
    parsed = tuple(_parse(QuestionContract, question) for question in questions)
    by_id = {question.question_id: question for question in parsed}
    if len(by_id) != len(parsed):
        raise ValueError("duplicate question IDs")
    return by_id


def evaluate_questions(questions: Iterable[QuestionContract | dict], *,
                       available: AvailableReferences) -> tuple[QuestionEvaluation, ...]:
    return tuple(evaluate_question(question, available=available)
                 for question in _questions_by_id(questions).values())


# Bounded detection prevents common numeric prose from bypassing typed gates.
_NUMERIC_TEXT = re.compile(
    r"(?:(?:risk|probability|chance|nnt|number needed to treat|absolute.{0,25}(?:effect|reduction)|"
    r"위험|확률|절대.{0,12}효과)[^\n.!?]{0,80}\d|\d+(?:\.\d+)?\s*%[^\n.!?]{0,40}(?:risk|위험|확률))",
    re.IGNORECASE,
)


def _numeric_reasons(claim, question, available):
    if claim.numeric_estimate is None:
        return ("untyped_numeric_estimate",) if _NUMERIC_TEXT.search(claim.text) else ()
    reasons = []
    compatible = ("PROGNOSIS", "DECISION") if claim.numeric_estimate.kind == "INDIVIDUAL_RISK" else ("TREATMENT_EFFECT", "DECISION")
    if question.type not in compatible:
        reasons.append("numeric_question_type_mismatch")
    support = claim.numeric_support
    if support is None:
        return (*reasons, "numeric_support_required")
    if not support.model_validated:
        reasons.append("numeric_model_not_validated")
    for name in ("validated_model_id", "validation_source_id", "calibration_source_id", "method_source_id"):
        source = getattr(support, name)
        if source is None:
            reasons.append(f"numeric_support_missing:{name}")
        elif name != "validated_model_id":
            if source not in available.evidence_ids:
                reasons.append(f"unknown_evidence_reference:{source}")
            elif source not in question.allowed_evidence_ids:
                reasons.append(f"disallowed_evidence_reference:{source}")
            if source not in claim.citations.evidence_ids:
                reasons.append(f"numeric_source_not_cited:{source}")
    required = _REQUIRED["PROGNOSIS" if claim.numeric_estimate.kind == "INDIVIDUAL_RISK" else "TREATMENT_EFFECT"]
    for name in required:
        declared, expected = getattr(support.scenario, name), getattr(question, name)
        if declared is None or expected is None:
            reasons.append(f"numeric_scenario_missing:{name}")
        elif declared != expected:
            reasons.append(f"numeric_scenario_mismatch:{name}")
    return tuple(reasons)


def validate_question_claims(questions: Iterable[QuestionContract | dict],
                             claims: Iterable[QuestionClaim | dict], *,
                             available: AvailableReferences) -> tuple[ClaimValidation, ...]:
    """Return one result per input claim; failed gates withhold only that claim.

    accepted means the structural gate passed, not that evidence proves the
    assertion. Explicit non-answer statuses remain visible with their own IDs.
    """
    by_id = _questions_by_id(questions)
    available = _parse(AvailableReferences, available)
    parsed = []
    for value in claims:
        try:
            parsed.append((_parse(QuestionClaim, value), None))
        except (ValueError, TypeError):
            raw = value.model_dump() if isinstance(value, StrictFrozenModel) else value
            raw = raw if isinstance(raw, dict) else {}
            identifiers = {}
            for name in ("claim_id", "question_id"):
                candidate = raw.get(name)
                identifiers[name] = candidate if isinstance(candidate, str) and re.fullmatch(r"\S(?:[\s\S]*\S)?", candidate) else None
            parsed.append((None, ClaimValidation(**identifiers, accepted=False,
                status="SYSTEM_FAILURE", reasons=("invalid_claim_contract",))))
    counts = Counter(claim.claim_id for claim, _ in parsed if claim)
    results = []
    for claim, failure in parsed:
        if failure is not None:
            results.append(failure)
            continue
        reasons = []
        status = claim.status
        question = by_id.get(claim.question_id)
        if counts[claim.claim_id] > 1:
            reasons.append("duplicate_claim_id")
            status = "CONFLICT"
        if question is None:
            reasons.append("unknown_question_id")
            status = "OUT_OF_SCOPE"
        else:
            refs = _reference_reasons(claim.citations, question, available)
            reasons.extend(refs)
            readiness = evaluate_question(question, available=available, decision_mode=claim.decision_mode)
            # Invalid references are never permitted, including in abstentions.
            if readiness.status == "CONFLICT":
                reasons.extend(readiness.reasons)
            if refs or readiness.status == "CONFLICT":
                status = "CONFLICT"
            elif claim.status in ("ANSWER", "CONDITIONAL"):
                if readiness.status == "MISSING_DATA":
                    reasons.extend(readiness.reasons)
                    status = "MISSING_DATA"
            numeric = _numeric_reasons(claim, question, available)
            if numeric:
                reasons.extend(numeric)
                if status not in ("CONFLICT", "OUT_OF_SCOPE"):
                    status = "MISSING_DATA"
        results.append(ClaimValidation(claim_id=claim.claim_id, question_id=claim.question_id,
            accepted=not reasons, status=status, reasons=tuple(dict.fromkeys(reasons))))
    return tuple(results)


class QuestionBundle(StrictFrozenModel):
    schema_version: Literal["case-questions/1"] = "case-questions/1"
    case_id: Identifier
    reviewer: Text
    reason: Text
    replaces: Identifier | None = None
    operation: Literal["upsert", "withdraw"] = "upsert"
    questions: tuple[QuestionContract, ...] = ()
    # Malformed individual claims are retained immutably for claim-local
    # SYSTEM_FAILURE results; they cannot suppress other questions/claims.
    claims: tuple[QuestionClaim | dict[str, JsonValue], ...] = ()
    additional_tests: tuple[AdditionalTestProposal, ...] = ()

    def model_post_init(self, __context):
        del __context
        claims = []
        for claim in self.claims:
            try:
                claims.append(_parse(QuestionClaim, claim))
            except (ValueError, TypeError):
                claims.append(deep_freeze(claim))
        object.__setattr__(self, "claims", tuple(claims))

    @model_validator(mode="after")
    def validate_operation(self):
        if self.operation == "withdraw":
            if not self.replaces or self.questions or self.claims or self.additional_tests:
                raise ValueError("withdrawal requires a target and no content")
        elif not self.questions:
            raise ValueError("an upsert requires explicit questions")
        return self


class QuestionEvent(StrictFrozenModel):
    schema_version: Literal["case-questions-event/1"] = "case-questions-event/1"
    event_id: Identifier
    version: int = Field(strict=True, ge=1)
    bundle: QuestionBundle
    available: AvailableReferences


class QuestionProjection(StrictFrozenModel):
    schema_version: Literal["case-questions-projection/1"] = "case-questions-projection/1"
    case_id: Identifier
    version: int = Field(strict=True, ge=0)
    event_count: int = Field(strict=True, ge=0)
    active: QuestionEvent | None = None
    evaluations: tuple[QuestionEvaluation, ...] = ()
    claim_results: tuple[ClaimValidation, ...] = ()


def _validate_bundle(bundle, available):
    by_id = _questions_by_id(bundle.questions)
    for question in by_id.values():
        if _reference_reasons(_allowed(question), None, available):
            raise ValueError("unknown question reference")
    proposals = set()
    for proposal in bundle.additional_tests:
        if proposal.proposal_id in proposals:
            raise ValueError("duplicate proposal ID")
        proposals.add(proposal.proposal_id)
        question = by_id.get(proposal.question_id)
        if question is None or _reference_reasons(proposal.citations, question, available):
            raise ValueError("unknown or disallowed additional test reference")


def _pin_available(bundle, available):
    requested = {kind: set() for kind in ("fact", "measurement", "evidence")}
    references = [_allowed(question) for question in bundle.questions]
    references.extend(proposal.citations for proposal in bundle.additional_tests)
    for claim in bundle.claims:
        if isinstance(claim, QuestionClaim):
            references.append(claim.citations)
            if claim.numeric_support:
                requested["evidence"].update(source for source in (
                    claim.numeric_support.validation_source_id, claim.numeric_support.calibration_source_id,
                    claim.numeric_support.method_source_id) if source is not None)
    for reference in references:
        for kind in requested:
            requested[kind].update(getattr(reference, f"{kind}_ids"))
    return AvailableReferences(**{f"{kind}_ids": frozenset(values & getattr(available, f"{kind}_ids"))
        for kind, values in requested.items()})


def _event_payload(event):
    return event.model_dump(mode="json", exclude={"event_id"})


class CaseQuestionService:
    """One immutable bundle chain per case, locked and validated on every write.

    Exact historical retries return their original event and never reactivate a
    withdrawn/replaced version. Loading without available replays the pinned
    reference snapshot; supply current sets to recheck withdrawn references.
    """
    def __init__(self, ledger: AppendOnlyJsonlLedger):
        self.ledger = ledger

    def history(self, case_id: str) -> tuple[QuestionEvent, ...]:
        case_id = _parse(QuestionProjection, {"case_id": case_id, "version": 0, "event_count": 0}).case_id
        events = []
        heads = {}
        for record in self.ledger.records():
            if record["kind"] != LEDGER_KIND:
                continue
            try:
                event = _parse(QuestionEvent, record["payload"])
                if record["natural_key"] != event.event_id or event.event_id != deterministic_id("case-questions", _event_payload(event)):
                    raise ValueError("invalid event hash")
                previous = heads.get(event.bundle.case_id)
                if event.version != (previous.version + 1 if previous else 1):
                    raise ValueError("invalid event version")
                if event.bundle.replaces != (previous.event_id if previous else None):
                    raise ValueError("invalid event predecessor")
                if event.bundle.operation == "withdraw":
                    if previous is None or previous.bundle.operation != "upsert" or event.available != AvailableReferences():
                        raise ValueError("invalid withdrawal")
                else:
                    _validate_bundle(event.bundle, event.available)
                    if _pin_available(event.bundle, event.available) != event.available:
                        raise ValueError("noncanonical reference snapshot")
                heads[event.bundle.case_id] = event
            except (ValueError, TypeError, AttributeError):
                raise RuntimeError("case questions ledger integrity check failed") from None
            if event.bundle.case_id == case_id:
                events.append(event)
        return tuple(events)

    def load(self, case_id: str, *, available: AvailableReferences | None = None) -> QuestionProjection:
        events = self.history(case_id)
        head = events[-1] if events else None
        active = head if head and head.bundle.operation == "upsert" else None
        known = _parse(AvailableReferences, available) if available is not None else active.available if active else AvailableReferences()
        return QuestionProjection(case_id=case_id, version=head.version if head else 0, event_count=len(events),
            active=active,
            evaluations=evaluate_questions(active.bundle.questions, available=known) if active else (),
            claim_results=validate_question_claims(active.bundle.questions, active.bundle.claims, available=known) if active else ())

    def save(self, bundle: QuestionBundle | dict, *, available: AvailableReferences) -> QuestionEvent:
        bundle = _parse(QuestionBundle, bundle)
        available = _parse(AvailableReferences, available)
        _validate_bundle(bundle, available)
        available = _pin_available(bundle, available)
        template = QuestionEvent(event_id="pending", version=1, bundle=bundle, available=available)
        request = _event_payload(template)
        request.pop("version")
        with self.ledger.key_lock("case-questions", bundle.case_id):
            events = self.history(bundle.case_id)
            for event in events:
                existing = _event_payload(event)
                existing.pop("version")
                if existing == request:
                    return event
            head = events[-1] if events else None
            if bundle.replaces != (head.event_id if head else None):
                raise ValueError("explicit replacement of the current questions head is required")
            if bundle.operation == "withdraw" and (head is None or head.bundle.operation != "upsert"):
                raise ValueError("only active questions can be withdrawn")
            payload = {**request, "version": head.version + 1 if head else 1}
            event = _parse(QuestionEvent, {**payload, "event_id": deterministic_id("case-questions", payload)})
            self.ledger.append_once(LEDGER_KIND, event.event_id,
                {**_event_payload(event), "event_id": event.event_id})
            return event

    def withdraw(self, case_id: str, event_id: str, *, reviewer: str, reason: str) -> QuestionEvent:
        events = self.history(case_id)
        target = next((event for event in events if event.event_id == event_id), None)
        if target is None or target.bundle.operation != "upsert":
            raise ValueError("unknown withdrawal target for case")
        return self.save(QuestionBundle(case_id=case_id, reviewer=reviewer, reason=reason,
            operation="withdraw", replaces=event_id), available=AvailableReferences())


def register_questions(case_subparsers):
    parser = case_subparsers.add_parser("questions", help="local bounded typed question review")
    parser.add_argument("--case", dest="case_id", required=True, metavar="CASE")
    parser.add_argument("--file", type=Path, help="audited QuestionBundle JSON; omit for status")
    parser.add_argument("--store", type=Path, default=argparse.SUPPRESS)


def _read_bundle(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("non-finite JSON value")

    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique_object, parse_constant=reject_constant)
    return _parse(QuestionBundle, value)


def _reviewed_references(ledger, case_id):
    from .cli import _latest_context

    try:
        context = _latest_context(ledger, case_id)
    except KeyError:
        context = {}
    # Share the runtime's reviewed and patient-eligibility boundary; preserve exact IDs.
    from .question_runtime import available_references
    refs = available_references(context)
    evidence = frozenset(item["evidence_id"] for item in context.get("evidence_cards", ()) if item.get("evidence_id"))
    return AvailableReferences(fact_ids=refs.fact_ids, measurement_ids=refs.measurement_ids, evidence_ids=evidence)


def _summary(projection):
    active = projection.active
    return {"schema_version": projection.schema_version, "case_id": projection.case_id,
        "version": projection.version, "event_count": projection.event_count,
        "event_id": active.event_id if active else None,
        "evaluations": [result.model_dump(mode="json") for result in projection.evaluations],
        "claim_results": [result.model_dump(mode="json") for result in projection.claim_results],
        "additional_test_ids": [test.proposal_id for test in active.bundle.additional_tests] if active else []}


def execute_questions(args) -> tuple[int, dict] | None:
    if getattr(args, "command", None) != "case" or getattr(args, "case_action", None) != "questions":
        return None
    try:
        ledger = AppendOnlyJsonlLedger(Path(getattr(args, "store", ".medic")) / "ledger.jsonl")
        service = CaseQuestionService(ledger)
        available = _reviewed_references(ledger, args.case_id)
        if getattr(args, "file", None) is not None:
            bundle = _read_bundle(args.file)
            if bundle.case_id != args.case_id:
                raise ValueError("questions bundle belongs to another case")
            service.save(bundle, available=available)
        return 0, _summary(service.load(args.case_id, available=available))
    except (ValueError, TypeError, OSError, RecursionError):
        return 2, {"error": "invalid_case_questions_input",
            "message": "Question input or references are invalid; no version appended."}
    except RuntimeError:
        return 2, {"error": "case_questions_integrity_error",
            "message": "Question history failed integrity validation; no version appended."}
