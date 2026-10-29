"""Bounded, offline patient context contracts and append-only review history.

Public entry point: CaseContextService(AppendOnlyJsonlLedger(...)). Profiles
and reviews reference existing facts; they neither create clinical facts nor
merge sources. Callers supply AvailableFact metadata from their trusted case
context. Only referenced metadata is pinned in each immutable event. No source
text is read, no dates/identity are inferred, and no provider is called.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .contracts import FrozenContract, NonEmptyStr, Sha256
from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, deterministic_id, sha256_digest


Missingness = Literal[
    "present", "explicit_negative", "not_provided", "not_measured", "not_assessable"
]
Classification = Literal[
    "observation", "clinician_assertion", "request", "procedure", "self_report",
    "actual_exposure", "derived_measure",
]
Category = Literal["profile", "input_review"]
LEDGER_KIND = "CASE_CONTEXT_VERSIONED"


class FactContext(FrozenContract):
    case_id: NonEmptyStr
    patient_id: NonEmptyStr | None = None
    episode_id: NonEmptyStr | None = None
    epoch_id: NonEmptyStr | None = None
    anatomy: NonEmptyStr | None = None
    laterality: Literal["left", "right", "bilateral", "midline", "unknown", "not_applicable"] | None = None


class AvailableFact(FrozenContract):
    fact_id: NonEmptyStr
    context: FactContext
    source_kind: Classification | Literal["prescription"] | None = None
    content_hash: Sha256 | None = None
    # Omit an absent optional pin to preserve hashes of existing ledger events.
    # This identifies reviewed value wording, independently of raw artifact bytes.
    value_sha256: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)


class AvailableContext(FrozenContract):
    """Strict --context JSON envelope; caller, not review content, vouches for IDs."""
    case_id: NonEmptyStr
    facts: tuple[AvailableFact, ...]


class ProfileItem(FrozenContract):
    name: NonEmptyStr
    value: NonEmptyStr | None = None
    missingness: Missingness
    fact_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def validate_value(self):
        if (self.missingness == "present") != (self.value is not None):
            raise ValueError("only present profile items carry a value")
        if len(set(self.fact_ids)) != len(self.fact_ids):
            raise ValueError("profile fact references must be unique")
        return self


class CaseProfile(FrozenContract):
    schema_version: Literal["case-profile/1"]
    case_id: NonEmptyStr
    patient_id: NonEmptyStr | None = None
    symptoms: tuple[ProfileItem, ...]
    function: tuple[ProfileItem, ...]
    activity: tuple[ProfileItem, ...]
    constraints: tuple[ProfileItem, ...]
    goals: tuple[ProfileItem, ...]
    priorities: tuple[ProfileItem, ...]

    @property
    def referenced_fact_ids(self) -> tuple[str, ...]:
        return tuple(sorted({fact_id for dimension in (
            self.symptoms, self.function, self.activity, self.constraints, self.goals, self.priorities
        ) for item in dimension for fact_id in item.fact_ids}))


class PartialDate(FrozenContract):
    role: Literal["authoring", "issuance", "documentation", "acquisition", "event",
                  "onset", "request", "procedure", "exposure"]
    precision: Literal["year", "month", "day", "unknown"]
    value: NonEmptyStr | None

    @model_validator(mode="after")
    def validate_precision(self):
        if self.precision == "unknown":
            if self.value is not None:
                raise ValueError("unknown date precision requires a null value")
            return self
        patterns = {"year": r"[0-9]{4}", "month": r"[0-9]{4}-[0-9]{2}",
                    "day": r"[0-9]{4}-[0-9]{2}-[0-9]{2}"}
        if self.value is None or not re.fullmatch(patterns[self.precision], self.value):
            raise ValueError("date value must match its declared precision")
        # Completing a date here validates its calendar bounds only. The original
        # partial string is retained; no imputed date is stored or returned.
        suffix = {"year": "-01-01", "month": "-01", "day": ""}[self.precision]
        try:
            date.fromisoformat(self.value + suffix)
        except ValueError:
            raise ValueError("invalid calendar date") from None
        return self


class ClinicalDuplicate(FrozenContract):
    relation: Literal["copied", "corrected", "true_repeat"]
    fact_id: NonEmptyStr


class SourceInputReview(FrozenContract):
    schema_version: Literal["source-input-review/1"]
    fact_id: NonEmptyStr
    context: FactContext
    classification: Classification
    missingness: Missingness
    dates: tuple[PartialDate, ...] = ()
    derived_from_ids: tuple[NonEmptyStr, ...] = ()
    duplicate: ClinicalDuplicate | None = None
    exposure_basis: Literal["documented_administration", "reported_taken", "documented_completion"] | None = None

    @model_validator(mode="after")
    def validate_semantics(self):
        if self.classification == "derived_measure" and not self.derived_from_ids:
            raise ValueError("derived measures require explicit lineage")
        if len(set(self.derived_from_ids)) != len(self.derived_from_ids):
            raise ValueError("derived references must be unique")
        if self.fact_id in self.derived_from_ids or (
            self.duplicate is not None and self.duplicate.fact_id == self.fact_id
        ):
            raise ValueError("source review cannot reference itself")
        if (self.classification == "actual_exposure") != (self.exposure_basis is not None):
            raise ValueError("actual exposure requires an explicit exposure basis")
        if self.classification == "actual_exposure" and self.missingness != "present":
            raise ValueError("missing or negated input cannot establish actual exposure")
        return self

    @property
    def actual_exposure(self) -> bool:
        return self.classification == "actual_exposure"

    @property
    def independent_evidence(self) -> bool:
        """Structural lineage flag only; not a clinical independence assessment."""
        return not self.derived_from_ids and not (
            self.duplicate and self.duplicate.relation in {"copied", "corrected"}
        )

    @property
    def referenced_fact_ids(self) -> tuple[str, ...]:
        return tuple(sorted({self.fact_id, *self.derived_from_ids,
                            *([self.duplicate.fact_id] if self.duplicate else [])}))


class ContextEvent(FrozenContract):
    schema_version: Literal["case-context/1"] = "case-context/1"
    event_id: NonEmptyStr
    case_id: NonEmptyStr
    category: Category
    subject_id: NonEmptyStr
    version: int = Field(strict=True, ge=1)
    operation: Literal["upsert", "withdraw"]
    reviewer: NonEmptyStr
    reason: NonEmptyStr
    replaces: NonEmptyStr | None = None
    profile: CaseProfile | None = None
    review: SourceInputReview | None = None
    available_facts: tuple[AvailableFact, ...] = ()

    @model_validator(mode="after")
    def validate_content(self):
        if self.operation == "withdraw":
            if self.profile is not None or self.review is not None or self.available_facts or not self.replaces:
                raise ValueError("withdrawal requires a target and cannot carry content")
        elif self.category == "profile":
            if self.profile is None or self.review is not None or self.subject_id != "profile":
                raise ValueError("profile event requires only a profile")
        elif self.review is None or self.profile is not None or self.subject_id != self.review.fact_id:
            raise ValueError("input review event requires only its subject review")
        return self


class CaseContextProjection(FrozenContract):
    schema_version: Literal["case-context-projection/1"] = "case-context-projection/1"
    case_id: NonEmptyStr
    profile: ContextEvent | None = None
    review_inputs: tuple[ContextEvent, ...] = ()
    event_count: int = Field(strict=True, ge=0)


def _validated_facts(case_id, content, available_facts):
    facts = tuple(AvailableFact.model_validate(item) for item in available_facts)
    by_id = {item.fact_id: item for item in facts}
    if len(by_id) != len(facts):
        raise ValueError("available fact IDs must be unambiguous")
    references = content.referenced_fact_ids
    if any(fact_id not in by_id for fact_id in references):
        raise ValueError("unknown fact reference")
    context = content.context if isinstance(content, SourceInputReview) else content
    if context.case_id != case_id:
        raise ValueError("content belongs to another case")
    selected = tuple(by_id[fact_id] for fact_id in references)
    if any(item.context.case_id != case_id for item in selected):
        raise ValueError("fact reference belongs to another case")
    patients = {item.context.patient_id for item in selected if item.context.patient_id is not None}
    if context.patient_id is not None:
        patients.add(context.patient_id)
    if len(patients) > 1:
        raise ValueError("fact reference belongs to another patient")
    if isinstance(content, SourceInputReview):
        primary = by_id[content.fact_id]
        for field in FactContext.model_fields:
            known, proposed = getattr(primary.context, field), getattr(content.context, field)
            if known is not None and proposed != known:
                raise ValueError("review context must preserve known primary fact identity")
        if primary.source_kind in {"request", "prescription"} and content.actual_exposure:
            raise ValueError("prescription or request is not evidence of actual exposure")
    return selected


def _key(event):
    return event.category, event.subject_id


class CaseContextService:
    """Serializes writers per case; replays/validates history on every operation.

    save_profile(case_id, profile, *, available_facts=(), reviewer, reason,
                 replaces=None) -> ContextEvent
    review_input(case_id, review, *, available_facts, reviewer, reason,
                 replaces=None) -> ContextEvent
    withdraw(case_id, event_id, *, reviewer, reason) -> ContextEvent
    status(case_id) -> CaseContextProjection; history(case_id) -> tuple[ContextEvent, ...]

    Changes require the current head ID, including after withdrawal. Exact
    retries are idempotent. Historical retry IDs never reactivate old content.
    """
    def __init__(self, ledger: AppendOnlyJsonlLedger):
        self.ledger = ledger

    def history(self, case_id: str) -> tuple[ContextEvent, ...]:
        case_id = FactContext(case_id=case_id).case_id
        events = []
        heads = {}
        for record in self.ledger.records():
            if record["kind"] != LEDGER_KIND:
                continue
            try:
                event = ContextEvent.model_validate(record["payload"])
                if event.event_id != deterministic_id("case-context", event.model_dump(
                    mode="json", exclude={"event_id"}
                )) or record["natural_key"] != event.event_id:
                    raise ValueError("invalid event hash")
                chain_key = (event.case_id, *_key(event))
                previous = heads.get(chain_key)
                if event.version != (previous.version + 1 if previous else 1):
                    raise ValueError("invalid event version")
                if event.replaces != (previous.event_id if previous else None):
                    raise ValueError("invalid event predecessor")
                if event.operation == "withdraw":
                    if previous is None or previous.operation != "upsert":
                        raise ValueError("invalid withdrawal target")
                else:
                    selected = _validated_facts(event.case_id, event.profile or event.review, event.available_facts)
                    if selected != event.available_facts:
                        raise ValueError("invalid reference snapshot")
                heads[chain_key] = event
            except (ValueError, TypeError, AttributeError):
                raise RuntimeError("case context ledger integrity check failed") from None
            if event.case_id == case_id:
                events.append(event)
        return tuple(events)

    def status(self, case_id: str) -> CaseContextProjection:
        events = self.history(case_id)
        heads = {_key(event): event for event in events}
        active = {key: event for key, event in heads.items() if event.operation == "upsert"}
        return CaseContextProjection(case_id=case_id, profile=active.get(("profile", "profile")),
            review_inputs=tuple(event for key, event in active.items() if key[0] == "input_review"),
            event_count=len(events))

    def save_profile(self, case_id: str, profile: CaseProfile | dict, *,
                     available_facts: Iterable[AvailableFact | dict] = (),
                     reviewer: str, reason: str, replaces: str | None = None) -> ContextEvent:
        profile = CaseProfile.model_validate(profile)
        facts = _validated_facts(case_id, profile, available_facts)
        return self._append(case_id, "profile", "profile", reviewer, reason, replaces,
                            profile=profile, available_facts=facts)

    def review_input(self, case_id: str, review: SourceInputReview | dict, *,
                     available_facts: Iterable[AvailableFact | dict], reviewer: str,
                     reason: str, replaces: str | None = None) -> ContextEvent:
        review = SourceInputReview.model_validate(review)
        facts = _validated_facts(case_id, review, available_facts)
        return self._append(case_id, "input_review", review.fact_id, reviewer, reason, replaces,
                            review=review, available_facts=facts)

    def withdraw(self, case_id: str, event_id: str, *, reviewer: str, reason: str) -> ContextEvent:
        events = self.history(case_id)
        target = next((event for event in events if event.event_id == event_id), None)
        if target is None or target.operation != "upsert":
            raise ValueError("unknown withdrawal target for case")
        return self._append(case_id, target.category, target.subject_id, reviewer, reason,
                            event_id, operation="withdraw")

    def _append(self, case_id, category, subject_id, reviewer, reason, replaces, *,
                profile=None, review=None, available_facts=(), operation="upsert"):
        # Validate audit and operation fields even before an idempotence lookup.
        template = ContextEvent(event_id="pending", case_id=case_id, category=category,
            subject_id=subject_id, version=1, operation=operation, reviewer=reviewer, reason=reason,
            replaces=replaces, profile=profile, review=review, available_facts=available_facts)
        request = template.model_dump(mode="json", exclude={"event_id", "version"})
        with self.ledger.key_lock("case-context", template.case_id):
            events = self.history(template.case_id)
            if operation == 'upsert':
                active = self.status(template.case_id)
                patients = {f.context.patient_id for f in available_facts if f.context.patient_id}
                if profile and profile.patient_id:
                    patients.add(profile.patient_id)
                if review and review.context.patient_id:
                    patients.add(review.context.patient_id)
                if active.profile and active.profile.profile.patient_id:
                    patients.add(active.profile.profile.patient_id)
                patients.update(e.review.context.patient_id for e in active.review_inputs if e.review.context.patient_id)
                if len(patients) > 1:
                    raise ValueError('patient identity conflicts with active case references; reconcile first')
            for event in events:
                if event.model_dump(mode="json", exclude={"event_id", "version"}) == request:
                    return event
            head = next((event for event in reversed(events) if _key(event) == _key(template)), None)
            if replaces != (head.event_id if head else None):
                raise ValueError("explicit replacement of the current head is required")
            if operation == "withdraw" and (head is None or head.operation != "upsert"):
                raise ValueError("only an active version can be withdrawn")
            payload = {**request, "version": head.version + 1 if head else 1}
            event = ContextEvent.model_validate({**payload, "event_id": deterministic_id("case-context", payload)})
            self.ledger.append_once(LEDGER_KIND, event.event_id, event.model_dump(mode="json"))
            return event


def _current_value_hashes(case_id: str, context: dict) -> dict[str, list[str | None]]:
    """Read only current reviewed facts in supplied context, never its extensions."""
    hashes: dict[str, list[str | None]] = {}

    def add(item):
        if not isinstance(item, dict):
            raise ValueError("current reviewed fact must be an object")
        fact_id = item.get("fact_id")
        if not fact_id or item.get("case_id", case_id) != case_id:
            return
        if not isinstance(fact_id, str):
            raise ValueError("current reviewed fact ID must be a string")
        value_hash = sha256_digest(canonical_json_bytes(item["value"])) if "value" in item else None
        hashes.setdefault(fact_id, []).append(value_hash)

    for item in context.get("fact_details", ()):
        if not isinstance(item, dict):
            raise ValueError("current reviewed fact must be an object")
        if item.get("user_confirmed") is True and item.get("confirmation_scope") != "UNREVIEWED_EXTRACTION":
            add(item)
    for review in context.get("imaging_reviews", ()):
        if not isinstance(review, dict):
            raise ValueError("current imaging review must be an object")
        if review.get("review_status") in {"USER_REVIEWED", "EXPERT_REVIEWED"}:
            for item in review.get("measurements", ()):
                add(item)
    return hashes


def _stale_references(event: ContextEvent, current: dict[str, list[str | None]]) -> list[dict[str, str]]:
    reasons = []
    for pinned in event.available_facts:
        values = current.get(pinned.fact_id, [])
        reason = None
        if not values:
            reason = "FACT_NOT_AVAILABLE"
        elif len(values) != 1:
            reason = "AMBIGUOUS_FACT_ID"
        elif pinned.value_sha256 is None:
            reason = "VALUE_HASH_UNPINNED"
        elif values[0] is None:
            reason = "CURRENT_VALUE_UNAVAILABLE"
        elif pinned.value_sha256 != values[0]:
            reason = "VALUE_CHANGED"
        if reason:
            reasons.append({"fact_id": pinned.fact_id, "reason": reason})
    return reasons


def enrich_context(ledger: AppendOnlyJsonlLedger, case_id: str, context: dict) -> dict:
    """Return an independent JSON copy with active M2 data under case_context.

    Existing confirmed_facts/fact_details are preserved byte-for-byte in JSON
    terms: no review fields are added to ConfirmedFactDetail or FactScope.
    case_context starts with the active ledger projection. review_inputs then
    includes only reviews whose complete reference set has a pinned value hash
    matching current reviewed fact_details or active imaging measurements in
    context. Stale event IDs and per-fact reason codes are exposed through
    stale_input_review_ids/stale_input_review_reasons. A linked profile is kept
    verbatim and flagged via profile_stale/profile_stale_reasons.

    Legacy unpinned events remain auditable but cannot establish runtime
    freshness. This function never calls the CLI, rewrites history, or uses a
    supplied case_context extension to validate itself.
    """
    if not isinstance(context, dict) or context.get("case_id", case_id) != case_id:
        raise ValueError("base context must belong to the requested case")
    result = json.loads(canonical_json_bytes(context))
    projection = CaseContextService(ledger).status(case_id)
    runtime = projection.model_dump(mode="json")
    current = _current_value_hashes(case_id, result)
    patient = projection.profile.profile.patient_id if projection.profile else None
    identities = {}
    for item in result.get('fact_details', ()):
        scope = item.get('scope', {})
        identities[item.get('fact_id')] = scope.get('patient_id') or scope.get('person_id')
    for review in result.get('imaging_reviews', ()):
        for item in review.get('measurements', ()):
            metadata = item.get('metadata', {})
            identities[item.get('fact_id')] = metadata.get('patient_id') or metadata.get('person_id')
    conflicts = {fid for fid, identity in identities.items() if fid and patient and identity and identity != patient}
    for event in projection.review_inputs:
        for fact in event.available_facts:
            identity = fact.context.patient_id
            if (patient and identity and patient != identity) or (identities.get(fact.fact_id) and identity and identities[fact.fact_id] != identity):
                conflicts.add(fact.fact_id)
    stale = {event.event_id: reasons for event in projection.review_inputs
             if (reasons := _stale_references(event, current))}
    for event in projection.review_inputs:
        for fact in event.available_facts:
            if fact.fact_id in conflicts:
                stale.setdefault(event.event_id, []).append({'fact_id': fact.fact_id, 'reason': 'PATIENT_ID_CONFLICT'})
    runtime['ineligible_fact_ids'] = sorted(conflicts)
    runtime['ineligible_fact_reasons'] = {fid: ['PATIENT_ID_CONFLICT'] for fid in conflicts}
    runtime['patient_identity_policy'] = 'KNOWN_CONFLICTS_ONLY_UNKNOWN_UNRESOLVED'
    runtime["review_inputs"] = [event.model_dump(mode="json") for event in projection.review_inputs
                                if event.event_id not in stale]
    runtime["stale_input_review_ids"] = list(stale)
    runtime["stale_input_review_reasons"] = stale
    profile_reasons = _stale_references(projection.profile, current) if projection.profile else []
    if projection.profile:
        profile_reasons += [{'fact_id': f.fact_id, 'reason': 'PATIENT_ID_CONFLICT'} for f in projection.profile.available_facts if f.fact_id in conflicts]
    runtime["profile_stale"] = bool(profile_reasons)
    runtime["profile_stale_reasons"] = profile_reasons
    result["case_context"] = runtime
    return result
