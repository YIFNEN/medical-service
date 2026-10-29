"""Argparse hooks for local M2 context. Parent integrates register/execute.

    register(subparsers)
    execute(args) -> (exit_code, JSON-safe summary) | None

    case status --case CASE [--store .medic]
    case profile --case CASE --file PROFILE.json --reviewer USER --reason TEXT
    case review-input --case CASE --file REVIEW.json --context CONTEXT.json ...

Mutations accept --replaces EVENT_ID; --withdraw EVENT_ID replaces --file.
CONTEXT.json is {"case_id": CASE, "facts": [AvailableFact, ...]}. Its IDs must
already exist in the case's current reviewed source facts or active imaging
measurement facts. It can fill unknown identity fields but cannot override
known source scope, source kind, or byte identity. The CLI lazily reads
medic.cli._latest_context; context extensions cannot whitelist themselves.
The CLI never prints raw profile/review text, reviewer,
reason, file paths, Pydantic input values, or exception details.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .case_context import AvailableContext, AvailableFact, CaseContextService, FactContext
from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, sha256_digest


def register(subparsers):
    case = subparsers.add_parser("case", help="local patient context and source input review")
    case.add_argument("--store", type=Path, default=Path(".medic"))
    actions = case.add_subparsers(dest="case_action", required=True)
    from .questions import register_questions
    register_questions(actions)
    for action in ("status", "profile", "review-input"):
        parser = actions.add_parser(action)
        parser.add_argument("--case", dest="case_id", required=True)
        parser.add_argument("--store", type=Path, default=argparse.SUPPRESS)
        if action == "status":
            continue
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--file", type=Path)
        mode.add_argument("--withdraw", metavar="EVENT_ID")
        parser.add_argument("--reviewer", required=True)
        parser.add_argument("--reason", required=True)
        parser.add_argument("--replaces", metavar="EVENT_ID")
        parser.add_argument("--context", type=Path)


def _read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("non-finite JSON value")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object, parse_constant=reject_constant)


def _summary(event):
    payload = {"event_id": event.event_id, "case_id": event.case_id,
               "category": event.category, "version": event.version,
               "operation": event.operation, "replaces": event.replaces}
    if event.review is not None:
        payload.update(fact_id=event.review.fact_id,
                       classification=event.review.classification,
                       actual_exposure=event.review.actual_exposure,
                       independent_evidence=event.review.independent_evidence)
    return payload


def _identity(case_id, *sources):
    """Adapt known source axes without inferring identity or merging anatomy."""
    values = {"case_id": case_id}
    aliases = {"person_id": "patient_id", "anatomical_site": "anatomy", "side": "laterality"}
    sides = {"r": "right", "rt": "right", "l": "left", "lt": "left", "b": "bilateral"}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source scope must be an object")
        for key, value in source.items():
            field = aliases.get(key, key)
            if field not in FactContext.model_fields or value is None:
                continue
            if isinstance(value, (list, tuple)) and field == "laterality":
                # Multiple study sides do not identify an individual measurement.
                if len(value) != 1:
                    continue
                value = value[0]
            if isinstance(value, str):
                value = value.strip()
                if not value or value.upper() in {"UNKNOWN", "UNREVIEWED", "NOT_PROVIDED"}:
                    continue
                if field == "laterality":
                    value = sides.get(value.lower(), value.lower())
            if field in values and values[field] != value:
                raise ValueError("known source identity fields conflict")
            values[field] = value
    return FactContext.model_validate(values)


def _reviewed_facts(ledger, case_id):
    # Import only at execution time: cli can itself import register/execute.
    from .cli import _latest_context

    try:
        context = _latest_context(ledger, case_id)
    except KeyError as exc:
        if not (len(exc.args) == 1 and isinstance(exc.args[0], str) and
                exc.args[0].startswith("no HIGH_IMPACT_FACTS_CONFIRMED data found for case ")):
            raise ValueError("reviewed case context could not be validated") from None
        context = {}
    if not isinstance(context, dict) or context.get("case_id", case_id) != case_id:
        raise ValueError("reviewed context belongs to another case")
    known = {}

    def add(item, *scopes):
        if not isinstance(item, dict):
            raise ValueError("reviewed fact must be an object")
        # Preserve actual IDs; legacy ID-less facts and display measurement IDs
        # are not given new identifiers or aliases here.
        if not item.get("fact_id"):
            return
        scope = item.get("scope") or {}
        raw_kind = item.get("source_kind", scope.get("source_kind"))
        kind = raw_kind.lower() if isinstance(raw_kind, str) else None
        if kind not in {"observation", "clinician_assertion", "request", "procedure", "self_report",
                        "actual_exposure", "derived_measure", "prescription"}:
            kind = None
        fact = AvailableFact(fact_id=item["fact_id"], context=_identity(case_id, *scopes, scope, item),
                             source_kind=kind, content_hash=item.get("content_hash"),
                             value_sha256=sha256_digest(canonical_json_bytes(item["value"]))
                             if "value" in item else None)
        if fact.fact_id in known:
            raise ValueError("reviewed fact IDs are ambiguous")
        known[fact.fact_id] = fact

    for item in context.get("fact_details", ()):
        if not isinstance(item, dict):
            raise ValueError("reviewed fact must be an object")
        if item.get("user_confirmed") is True and item.get("confirmation_scope") != "UNREVIEWED_EXTRACTION":
            add(item)
    for review in context.get("imaging_reviews", ()):
        if review.get("review_status") not in {"USER_REVIEWED", "EXPERT_REVIEWED"}:
            continue
        for item in review.get("measurements", ()):
            add(item, review.get("conditions") or {}, item.get("metadata") or {})
    return known


def _available_facts(ledger, case_id, context_path):
    known = _reviewed_facts(ledger, case_id)
    if context_path is not None:
        supplied = AvailableContext.model_validate(_read_json(context_path))
        if supplied.case_id != case_id or len({f.fact_id for f in supplied.facts}) != len(supplied.facts):
            raise ValueError("available context must identify unique facts in this case")
        for overlay in supplied.facts:
            actual = known.get(overlay.fact_id)
            if actual is None or overlay.context.case_id != case_id:
                raise ValueError("supplied context contains an unknown fact for this case")
            identity = actual.context.model_dump()
            for field, value in overlay.context.model_dump().items():
                if value is None:
                    continue
                if identity[field] is not None and identity[field] != value:
                    raise ValueError("supplied context conflicts with known source scope")
                identity[field] = value
            if any(getattr(overlay, field) is not None and getattr(overlay, field) != getattr(actual, field)
                   for field in ("source_kind", "content_hash", "value_sha256")):
                raise ValueError("supplied context cannot override source metadata")
            known[overlay.fact_id] = AvailableFact(fact_id=actual.fact_id,
                context=FactContext.model_validate(identity), source_kind=actual.source_kind,
                content_hash=actual.content_hash, value_sha256=actual.value_sha256)
    return tuple(known[key] for key in sorted(known))


def execute(args):
    if getattr(args, "command", None) != "case":
        return None
    try:
        from .questions import execute_questions
        handled = execute_questions(args)
        if handled is not None:
            return handled
        service = CaseContextService(AppendOnlyJsonlLedger(Path(args.store) / "ledger.jsonl"))
        if args.case_action == "status":
            status = service.status(args.case_id)
            return 0, {"case_id": status.case_id, "schema_version": status.schema_version,
                       "event_count": status.event_count,
                       "profile": _summary(status.profile) if status.profile else None,
                       "review_inputs": [_summary(event) for event in status.review_inputs]}
        if args.case_action not in {"profile", "review-input"}:
            raise ValueError("unknown case action")
        if args.withdraw:
            if args.replaces is not None or args.context is not None:
                raise ValueError("withdraw cannot carry replacement or context")
            target = next((e for e in service.history(args.case_id) if e.event_id == args.withdraw), None)
            category = "profile" if args.case_action == "profile" else "input_review"
            if target is None or target.category != category:
                raise ValueError("withdrawal target has the wrong category")
            event = service.withdraw(args.case_id, args.withdraw, reviewer=args.reviewer, reason=args.reason)
        else:
            facts = _available_facts(service.ledger, args.case_id, args.context)
            method = service.save_profile if args.case_action == "profile" else service.review_input
            event = method(args.case_id, _read_json(args.file), available_facts=facts,
                           reviewer=args.reviewer, reason=args.reason, replaces=args.replaces)
        return 0, _summary(event)
    except (ValueError, TypeError, OSError, RecursionError):
        return 2, {"error": "invalid_case_context_input",
                   "message": "Case context input or references are invalid; no version appended."}
    except RuntimeError:
        return 2, {"error": "case_context_integrity_error",
                   "message": "Case context history failed integrity validation; no version appended."}
