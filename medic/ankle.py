from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, ValidationError

from .compiler import CompileResult
from .contracts import (
    AnkleStateVersion,
    CandidateFact,
    ClaimReadiness,
    ClaimView,
    ClinicalEventVersion,
    FeatureValue,
    FeatureVersion,
    NonEmptyStr,
    VerificationDecision,
    VerificationStatus,
)
from .ledger import (
    AppendOnlyJsonlLedger,
    canonical_json_bytes,
    deterministic_id,
    sha256_digest,
)


VERIFICATION_COMPONENT_VERSION = "verification-queue/0.1.0"
VERIFICATION_POLICY_VERSION = "high-impact-source-review/0.1.0"
EVENT_COMPONENT_VERSION = "clinical-event-projector/0.1.0"
EVENT_POLICY_VERSION = "clinical-event-promotion/0.1.0"
STATE_COMPONENT_VERSION = "ankle-state-materializer/0.1.0"
STATE_POLICY_VERSION = "ankle-state-routing/0.1.0"
FEATURE_COMPONENT_VERSION = "ankle-feature-builder/0.1.0"
FEATURE_POLICY_VERSION = "structural-coverage-feature/0.1.0"
CLAIM_COMPONENT_VERSION = "ankle-claim-view/0.1.0"
CLAIM_POLICY_VERSION = "post-orif-structural-readiness/0.1.0"
CLAIM_CODE = "ANKLE_POST_ORIF_STRUCTURAL_STATE"


class ReviewEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: NonEmptyStr
    status: VerificationStatus
    corrected_value: JsonValue | None = None
    corrected_effective_at: AwareDatetime | None = None
    corrected_side: NonEmptyStr | None = None
    corrected_episode_id: NonEmptyStr | None = None
    corrected_epoch_id: NonEmptyStr | None = None
    replaces_decision_id: NonEmptyStr | None = None


class ReviewBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: NonEmptyStr
    decided_at: AwareDatetime
    decisions: tuple[ReviewEntry, ...]


@dataclass(frozen=True)
class MaterializationResult:
    user_decisions: tuple[VerificationDecision, ...]
    system_decisions: tuple[VerificationDecision, ...]
    events: tuple[ClinicalEventVersion, ...]
    new_events: tuple[ClinicalEventVersion, ...]
    state: AnkleStateVersion
    feature: FeatureVersion
    claim_view: ClaimView


class ClinicalAnkleProjector:
    def __init__(self, ledger: AppendOnlyJsonlLedger) -> None:
        self.ledger = ledger

    def review_and_materialize(
        self,
        compiled: CompileResult,
        decisions_path: Path,
    ) -> MaterializationResult:
        with self.ledger.key_lock("case", compiled.artifact.case_id):
            return self._review_locked(compiled, decisions_path)

    def _review_locked(
        self,
        compiled: CompileResult,
        decisions_path: Path,
    ) -> MaterializationResult:
        try:
            bundle = ReviewBundle.model_validate_json(Path(decisions_path).read_bytes())
        except ValidationError as exc:
            raise ValueError(f"review bundle schema validation failed: {exc}") from exc
        entries = {entry.candidate_id: entry for entry in bundle.decisions}
        if len(entries) != len(bundle.decisions):
            raise ValueError("review bundle contains duplicate candidate_id decisions")
        review_candidate_ids = {
            candidate.candidate_id for candidate in compiled.review_queue
        }
        if not set(entries).issubset(review_candidate_ids):
            raise ValueError("review bundle contains a candidate outside the high-impact queue")

        prior_decisions = self._latest_decisions()
        unresolved = {
            candidate.candidate_id
            for candidate in compiled.review_queue
            if candidate.candidate_id not in prior_decisions
            or prior_decisions[candidate.candidate_id].component_version
            != VERIFICATION_COMPONENT_VERSION
            or prior_decisions[candidate.candidate_id].policy_version
            != VERIFICATION_POLICY_VERSION
        }
        if not unresolved.issubset(entries):
            raise ValueError("review bundle must decide every unresolved high-impact candidate")

        existing_event_ids = {
            record["payload"]["event"]["event_id"]
            for record in self.ledger.records()
            if record["kind"] == "CLINICAL_EVENT_PROMOTED"
        }
        user_decisions: list[VerificationDecision] = []
        system_decisions: list[VerificationDecision] = []
        events: list[ClinicalEventVersion] = []
        batch_entries: list[tuple[str, str, dict[str, Any]]] = []
        block_by_id = {block.block_id: block for block in compiled.blocks}
        for candidate in compiled.candidates:
            if candidate.high_impact:
                prior = prior_decisions.get(candidate.candidate_id)
                entry = entries.get(candidate.candidate_id)
                prior_is_current = (
                    prior is not None
                    and prior.component_version == VERIFICATION_COMPONENT_VERSION
                    and prior.policy_version == VERIFICATION_POLICY_VERSION
                )
                if prior is not None and not prior_is_current:
                    assert entry is not None
                    entry = entry.model_copy(update={"replaces_decision_id": prior.decision_id})
                    decision = self._decision(candidate, entry, bundle.reviewer, bundle.decided_at)
                elif prior is not None and entry is None:
                    decision = prior
                elif prior is not None and entry is not None and entry.replaces_decision_id is None:
                    if (
                        entry.status is not prior.status
                        or canonical_json_bytes(entry.corrected_value)
                        != canonical_json_bytes(prior.corrected_value)
                        or entry.corrected_effective_at != prior.corrected_effective_at
                        or entry.corrected_side != prior.corrected_side
                        or entry.corrected_episode_id != prior.corrected_episode_id
                        or entry.corrected_epoch_id != prior.corrected_epoch_id
                    ):
                        raise ValueError("a changed verification requires replaces_decision_id")
                    decision = prior
                else:
                    if prior is None and entry is not None and entry.replaces_decision_id is not None:
                        raise ValueError("initial verification cannot replace a decision")
                    if prior is not None and entry.replaces_decision_id != prior.decision_id:
                        raise ValueError("verification revision must replace the latest decision")
                    assert entry is not None
                    decision = self._decision(candidate, entry, bundle.reviewer, bundle.decided_at)
            else:
                prior = prior_decisions.get(candidate.candidate_id)
                if (
                    prior is not None
                    and prior.component_version == VERIFICATION_COMPONENT_VERSION
                    and prior.policy_version == VERIFICATION_POLICY_VERSION
                ):
                    decision = prior
                else:
                    entry = ReviewEntry(
                        candidate_id=candidate.candidate_id,
                        status=VerificationStatus.CONFIRMED,
                        replaces_decision_id=prior.decision_id if prior else None,
                    )
                    decision = self._decision(
                        candidate,
                        entry,
                        "SYSTEM_LOW_IMPACT_POLICY",
                        datetime(1970, 1, 1, tzinfo=UTC),
                    )
            batch_entries.append(
                (
                    "VERIFICATION_DECIDED",
                    decision.decision_id,
                    {"decision": decision.model_dump(mode="json")},
                )
            )
            if candidate.high_impact:
                user_decisions.append(decision)
            else:
                system_decisions.append(decision)
            if decision.status is VerificationStatus.UNKNOWN:
                continue
            event = self._event(compiled, candidate, decision, block_by_id)
            batch_entries.append(
                (
                    "CLINICAL_EVENT_PROMOTED",
                    event.event_id,
                    {"event": event.model_dump(mode="json")},
                )
            )
            events.append(event)

        self.ledger.append_batch_once(tuple(batch_entries))
        new_events = tuple(event for event in events if event.event_id not in existing_event_ids)
        state, feature, claim_view = self._materialize(compiled.artifact.case_id)
        return MaterializationResult(
            user_decisions=tuple(user_decisions),
            system_decisions=tuple(system_decisions),
            events=tuple(events),
            new_events=new_events,
            state=state,
            feature=feature,
            claim_view=claim_view,
        )

    def _decision(
        self,
        candidate: CandidateFact,
        entry: ReviewEntry,
        reviewer: str,
        decided_at: datetime,
    ) -> VerificationDecision:
        decision_payload = {
            "candidate_id": candidate.candidate_id,
            "status": entry.status.value,
            "corrected_value": entry.corrected_value,
            "corrected_effective_at": (
                entry.corrected_effective_at.isoformat() if entry.corrected_effective_at else None
            ),
            "replaces_decision_id": entry.replaces_decision_id,
            "corrected_side": entry.corrected_side,
            "corrected_episode_id": entry.corrected_episode_id,
            "corrected_epoch_id": entry.corrected_epoch_id,
            "component_version": VERIFICATION_COMPONENT_VERSION,
            "policy_version": VERIFICATION_POLICY_VERSION,
        }
        resolved_side = entry.corrected_side or candidate.side
        if entry.status is VerificationStatus.CORRECTED and isinstance(
            entry.corrected_value, dict
        ):
            value_side = entry.corrected_value.get("side")
            if value_side is not None and value_side != resolved_side:
                raise ValueError("corrected value side must equal the resolved corrected_side")
        decision_hash = sha256_digest(canonical_json_bytes(decision_payload))
        return VerificationDecision(
            decision_id=deterministic_id("decision", decision_payload),
            candidate_id=candidate.candidate_id,
            status=entry.status,
            decided_at=decided_at,
            reviewer=reviewer,
            corrected_value=entry.corrected_value,
            corrected_effective_at=entry.corrected_effective_at,
            corrected_side=entry.corrected_side,
            corrected_episode_id=entry.corrected_episode_id,
            corrected_epoch_id=entry.corrected_epoch_id,
            replaces_decision_id=entry.replaces_decision_id,
            source_locators=candidate.source_locators,
            input_hash=decision_hash,
            component_version=VERIFICATION_COMPONENT_VERSION,
            policy_version=VERIFICATION_POLICY_VERSION,
            dependency_ids=(candidate.candidate_id,),
        )

    def _latest_decisions(self) -> dict[str, VerificationDecision]:
        by_candidate: dict[str, list[VerificationDecision]] = {}
        for record in self.ledger.records():
            if record["kind"] != "VERIFICATION_DECIDED":
                continue
            decision = VerificationDecision.model_validate(record["payload"]["decision"])
            by_candidate.setdefault(decision.candidate_id, []).append(decision)
        latest: dict[str, VerificationDecision] = {}
        for candidate_id, decisions in by_candidate.items():
            by_id = {decision.decision_id: decision for decision in decisions}
            children: dict[str, list[str]] = {}
            for decision in decisions:
                parent = decision.replaces_decision_id
                if parent is None:
                    continue
                if parent not in by_id:
                    raise RuntimeError("verification chain has a missing predecessor")
                children.setdefault(parent, []).append(decision.decision_id)
            if any(len(child_ids) != 1 for child_ids in children.values()):
                raise RuntimeError("verification chain contains a revision fork")
            leaves = [decision for decision in decisions if decision.decision_id not in children]
            if len(leaves) != 1:
                raise RuntimeError("verification chain does not have exactly one active leaf")
            latest[candidate_id] = leaves[0]
        return latest

    def _event(
        self,
        compiled: CompileResult,
        candidate: CandidateFact,
        decision: VerificationDecision,
        block_by_id: dict[str, Any],
    ) -> ClinicalEventVersion:
        prior_events = [
            ClinicalEventVersion.model_validate(record["payload"]["event"])
            for record in self.ledger.records()
            if record["kind"] == "CLINICAL_EVENT_PROMOTED"
            and record["payload"]["event"]["candidate_id"] == candidate.candidate_id
        ]
        existing = next(
            (
                event
                for event in prior_events
                if event.verification_decision_id == decision.decision_id
                and event.component_version == EVENT_COMPONENT_VERSION
                and event.policy_version == EVENT_POLICY_VERSION
            ),
            None,
        )
        if existing is not None:
            return existing
        version = max((event.version for event in prior_events), default=0) + 1
        predecessor = max(prior_events, key=lambda event: event.version) if prior_events else None
        value = (
            decision.corrected_value
            if decision.status is VerificationStatus.CORRECTED
            else candidate.normalized_value
        )
        block = block_by_id[candidate.dependency_ids[0]]
        event_payload = {
            "candidate_id": candidate.candidate_id,
            "decision_id": decision.decision_id,
            "version": version,
            "value": value,
            "component_version": EVENT_COMPONENT_VERSION,
            "policy_version": EVENT_POLICY_VERSION,
        }
        return ClinicalEventVersion(
            event_id=deterministic_id("event", event_payload),
            case_id=compiled.artifact.case_id,
            candidate_id=candidate.candidate_id,
            version=version,
            semantic_type=candidate.semantic_type,
            field_name=candidate.field_name,
            side=decision.corrected_side or candidate.side,
            episode_id=decision.corrected_episode_id or candidate.episode_id,
            epoch_id=decision.corrected_epoch_id or candidate.epoch_id,
            effective_at=decision.corrected_effective_at or block.raw_payload["effective_at"],
            value=value,
            verification_decision_id=decision.decision_id,
            supersedes_event_id=predecessor.event_id if predecessor else None,
            source_locators=candidate.source_locators,
            input_hash=sha256_digest(canonical_json_bytes(event_payload)),
            component_version=EVENT_COMPONENT_VERSION,
            policy_version=EVENT_POLICY_VERSION,
            dependency_ids=(candidate.candidate_id, decision.decision_id),
        )

    def _materialize(
        self, case_id: str
    ) -> tuple[AnkleStateVersion, FeatureVersion, ClaimView]:
        records = self.ledger.records()
        all_events = [
            ClinicalEventVersion.model_validate(record["payload"]["event"])
            for record in records
            if record["kind"] == "CLINICAL_EVENT_PROMOTED"
            and record["payload"]["event"]["case_id"] == case_id
        ]
        latest_decisions = self._latest_decisions()
        latest_by_candidate: dict[str, ClinicalEventVersion] = {}
        for event in all_events:
            latest_decision = latest_decisions.get(event.candidate_id)
            if (
                latest_decision is None
                or latest_decision.status is VerificationStatus.UNKNOWN
                or event.verification_decision_id != latest_decision.decision_id
            ):
                continue
            current = latest_by_candidate.get(event.candidate_id)
            if current is None or event.version > current.version:
                latest_by_candidate[event.candidate_id] = event
        all_active_events = tuple(
            sorted(
                latest_by_candidate.values(),
                key=lambda event: (event.effective_at or datetime.min.replace(tzinfo=UTC), event.event_id),
            )
        )
        anchor = next(
            (event for event in all_active_events if event.field_name == "index_injury"),
            all_active_events[0] if all_active_events else None,
        )
        if anchor is None:
            raise ValueError("no active verified event is available for ankle materialization")
        active_events = tuple(
            event
            for event in all_active_events
            if event.side == anchor.side
            and event.episode_id == anchor.episode_id
            and event.epoch_id == anchor.epoch_id
        )
        event_ids = tuple(event.event_id for event in active_events)
        source_locators = tuple(
            locator for event in active_events for locator in event.source_locators
        )
        dependency_hash = sha256_digest(
            canonical_json_bytes(
                {
                    "event_ids": event_ids,
                    "component_version": STATE_COMPONENT_VERSION,
                    "policy_version": STATE_POLICY_VERSION,
                }
            )
        )

        existing_states = [
            AnkleStateVersion.model_validate(record["payload"]["state"])
            for record in records
            if record["kind"] == "ANKLE_STATE_MATERIALIZED"
            and record["payload"]["state"]["case_id"] == case_id
        ]
        existing_state = next(
            (
                state
                for state in existing_states
                if state.dependency_hash == dependency_hash
                and state.component_version == STATE_COMPONENT_VERSION
                and state.policy_version == STATE_POLICY_VERSION
            ),
            None,
        )
        if existing_state is not None:
            feature = next(
                (
                    FeatureVersion.model_validate(record["payload"]["feature"])
                    for record in records
                    if record["kind"] == "FEATURE_MATERIALIZED"
                    and record["payload"]["feature"]["dependency_ids"][0]
                    == existing_state.state_id
                    and record["payload"]["feature"]["component_version"]
                    == FEATURE_COMPONENT_VERSION
                    and record["payload"]["feature"]["policy_version"]
                    == FEATURE_POLICY_VERSION
                ),
                None,
            )
            if feature is None:
                feature = self._feature(existing_state, active_events, source_locators)
                self.ledger.append_once(
                    "FEATURE_MATERIALIZED",
                    feature.feature_id,
                    {"feature": feature.model_dump(mode="json")},
                )
            claim_view = next(
                (
                    ClaimView.model_validate(record["payload"]["claim_view"])
                    for record in records
                    if record["kind"] == "CLAIM_VIEW_MATERIALIZED"
                    and record["payload"]["claim_view"]["state_id"] == existing_state.state_id
                    and record["payload"]["claim_view"]["component_version"]
                    == CLAIM_COMPONENT_VERSION
                    and record["payload"]["claim_view"]["policy_version"]
                    == CLAIM_POLICY_VERSION
                ),
                None,
            )
            if claim_view is None:
                claim_view = self._claim_view(existing_state, active_events, source_locators)
                self.ledger.append_once(
                    "CLAIM_VIEW_MATERIALIZED",
                    claim_view.claim_view_id,
                    {"claim_view": claim_view.model_dump(mode="json")},
                )
            return existing_state, feature, claim_view

        values = [(event, event.value) for event in active_events if isinstance(event.value, dict)]
        side = next((value.get("side") for _event, value in values if value.get("side")), "UNKNOWN")
        fixation: dict[str, JsonValue] = {}
        symptoms: dict[str, JsonValue] = {}
        for event, value in values:
            if event.semantic_type == "PROCEDURE":
                fixation.update(value)
            if event.semantic_type in {"REQUEST", "EXPOSURE", "SYMPTOM"}:
                symptoms[event.field_name] = value
        previous = max(existing_states, key=lambda state: state.version) if existing_states else None
        version = (previous.version + 1) if previous else 1
        state_id = deterministic_id(
            "ankle-state",
            {
                "case_id": case_id,
                "dependency_hash": dependency_hash,
                "component_version": STATE_COMPONENT_VERSION,
                "policy_version": STATE_POLICY_VERSION,
            },
        )
        state_dependencies = event_ids + ((previous.state_id,) if previous else ())
        state = AnkleStateVersion(
            state_id=state_id,
            case_id=case_id,
            version=version,
            side=anchor.side if side == "UNKNOWN" else side,
            episode_id=anchor.episode_id,
            epoch_id=anchor.epoch_id,
            event_ids=event_ids,
            fixation_construct=fixation,
            healing_units=self._healing_units(active_events),
            symptoms_function_load=symptoms,
            dependency_hash=dependency_hash,
            source_locators=source_locators,
            input_hash=dependency_hash,
            component_version=STATE_COMPONENT_VERSION,
            policy_version=STATE_POLICY_VERSION,
            dependency_ids=state_dependencies,
        )
        self.ledger.append_once(
            "ANKLE_STATE_MATERIALIZED", state.state_id, {"state": state.model_dump(mode="json")}
        )
        feature = self._feature(state, active_events, source_locators)
        self.ledger.append_once(
            "FEATURE_MATERIALIZED",
            feature.feature_id,
            {"feature": feature.model_dump(mode="json")},
        )
        claim_view = self._claim_view(state, active_events, source_locators)
        self.ledger.append_once(
            "CLAIM_VIEW_MATERIALIZED",
            claim_view.claim_view_id,
            {"claim_view": claim_view.model_dump(mode="json")},
        )
        return state, feature, claim_view

    @staticmethod
    def _healing_units(events: tuple[ClinicalEventVersion, ...]) -> dict[str, JsonValue]:
        trimalleolar = any(
            event.field_name == "index_injury"
            and isinstance(event.value, dict)
            and event.value.get("pattern") == "TRIMALLEOLAR_FRACTURE_DISLOCATION"
            for event in events
        )
        if not trimalleolar:
            return {}
        return {
            structure: {"state": "NOT_ASSESSABLE", "source": "INDEX_INJURY_PATTERN"}
            for structure in ("MEDIAL", "LATERAL", "POSTERIOR")
        }

    def _feature(
        self,
        state: AnkleStateVersion,
        events: tuple[ClinicalEventVersion, ...],
        source_locators: tuple[Any, ...],
    ) -> FeatureVersion:
        feature_id = deterministic_id(
            "feature",
            {
                "state_id": state.state_id,
                "definition": "structural-coverage/v1",
                "component_version": FEATURE_COMPONENT_VERSION,
                "policy_version": FEATURE_POLICY_VERSION,
            },
        )
        value = {
            "eligible_event_count": len(events),
            "has_injury": any(event.semantic_type == "INJURY" for event in events),
            "has_procedure": any(event.semantic_type == "PROCEDURE" for event in events),
            "has_alignment": any(event.semantic_type == "OBSERVATION" for event in events),
        }
        payload = FeatureValue(
            feature_id=feature_id,
            definition_id="ANKLE_POST_ORIF_STRUCTURAL_COVERAGE",
            definition_version="1.0.0",
            input_fact_or_roi_ids=tuple(event.event_id for event in events),
            source_locators=source_locators,
            observed_or_derived="DERIVED",
            value=value,
            method_or_algorithm_version=FEATURE_COMPONENT_VERSION,
            qc_status="PASS",
            allowed_uses=(CLAIM_CODE,),
            dependency_hash=state.dependency_hash,
        )
        existing_versions = [
            FeatureVersion.model_validate(record["payload"]["feature"]).version
            for record in self.ledger.records()
            if record["kind"] == "FEATURE_MATERIALIZED"
            and record["payload"]["feature"]["case_id"] == state.case_id
            and record["payload"]["feature"]["name"]
            == "ANKLE_POST_ORIF_STRUCTURAL_COVERAGE"
        ]
        return FeatureVersion(
            feature_id=feature_id,
            case_id=state.case_id,
            version=max(existing_versions, default=0) + 1,
            name="ANKLE_POST_ORIF_STRUCTURAL_COVERAGE",
            payload=payload,
            dependency_hash=state.dependency_hash,
            source_locators=source_locators,
            input_hash=state.dependency_hash,
            component_version=FEATURE_COMPONENT_VERSION,
            policy_version=FEATURE_POLICY_VERSION,
            dependency_ids=(state.state_id,) + tuple(event.event_id for event in events),
        )

    def _claim_view(
        self,
        state: AnkleStateVersion,
        events: tuple[ClinicalEventVersion, ...],
        source_locators: tuple[Any, ...],
    ) -> ClaimView:
        required = {"index_injury", "index_orif", "postoperative_alignment"}
        qualified: dict[str, ClinicalEventVersion] = {}
        for event in events:
            if not isinstance(event.value, dict):
                continue
            if (
                event.field_name == "index_injury"
                and event.semantic_type == "INJURY"
                and event.value.get("pattern") == "TRIMALLEOLAR_FRACTURE_DISLOCATION"
            ):
                qualified[event.field_name] = event
            elif (
                event.field_name == "index_orif"
                and event.semantic_type == "PROCEDURE"
                and event.value.get("procedure") == "ORIF"
            ):
                qualified[event.field_name] = event
            elif (
                event.field_name == "postoperative_alignment"
                and event.semantic_type == "OBSERVATION"
                and event.value.get("mortise") is not None
                and event.value.get("source_modality") == "XR"
            ):
                qualified[event.field_name] = event
        missing = tuple(sorted(required - set(qualified)))
        readiness = ClaimReadiness.READY if not missing else ClaimReadiness.NOT_READY
        support = tuple(
            event.event_id
            for event in events
            if event.field_name in qualified
            and qualified[event.field_name].event_id == event.event_id
            and not (
                event.field_name == "postoperative_alignment"
                and isinstance(event.value, dict)
                and event.value.get("mortise") != "MAINTAINED"
            )
        )
        contradicting = tuple(
            event.event_id
            for event in events
            if event.field_name == "postoperative_alignment"
            and isinstance(event.value, dict)
            and event.value.get("mortise") != "MAINTAINED"
        )
        claim_id = deterministic_id(
            "claim-view",
            {
                "state_id": state.state_id,
                "claim": CLAIM_CODE,
                "component_version": CLAIM_COMPONENT_VERSION,
                "policy_version": CLAIM_POLICY_VERSION,
            },
        )
        return ClaimView(
            claim_view_id=claim_id,
            case_id=state.case_id,
            claim_code=CLAIM_CODE,
            state_id=state.state_id,
            readiness=readiness,
            eligible_event_ids=tuple(event.event_id for event in events),
            supporting_event_ids=support,
            contradicting_event_ids=contradicting,
            missing_requirements=missing,
            uncertainty_notes=(),
            dependency_hash=state.dependency_hash,
            source_locators=source_locators,
            input_hash=state.dependency_hash,
            component_version=CLAIM_COMPONENT_VERSION,
            policy_version=CLAIM_POLICY_VERSION,
            dependency_ids=(state.state_id,) + tuple(event.event_id for event in events),
        )
