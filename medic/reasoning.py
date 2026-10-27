from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from .contracts import (
    AnalysisRun,
    AnkleStateVersion,
    ClaimView,
    FeatureVersion,
    JsonValue,
    SnapshotManifest,
)
from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, deterministic_id, sha256_digest


COMPONENT_VERSION = "reasoning-service/0.1.0"
POLICY_VERSION = "synthetic-non-clinical-only/0.1.0"


class Reasoner(Protocol):
    name: str
    version: str

    def reason(self, claim_view: ClaimView) -> dict[str, JsonValue]: ...


class NonClinicalReasoningOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    medical_conclusion_generated: Literal[False]
    claim_code: str
    readiness: str
    eligible_event_count: int
    supporting_event_count: int
    contradicting_event_count: int
    missing_requirements: list[str]
    notice: str


class SyntheticReasoner:
    name = "synthetic"
    version = "1.0.0"

    def reason(self, claim_view: ClaimView) -> dict[str, JsonValue]:
        return expected_non_clinical_output(claim_view)


def expected_non_clinical_output(claim_view: ClaimView) -> dict[str, JsonValue]:
    return {
        "medical_conclusion_generated": False,
        "claim_code": claim_view.claim_code,
        "readiness": claim_view.readiness.value,
        "eligible_event_count": len(claim_view.eligible_event_ids),
        "supporting_event_count": len(claim_view.supporting_event_ids),
        "contradicting_event_count": len(claim_view.contradicting_event_ids),
        "missing_requirements": list(claim_view.missing_requirements),
        "notice": "Synthetic pipeline verification only; no diagnosis, prognosis, or treatment advice.",
    }


class ReasoningService:
    def __init__(self, ledger: AppendOnlyJsonlLedger) -> None:
        self.ledger = ledger

    def analyze(self, case_id: str, claim_code: str, reasoner: Reasoner) -> AnalysisRun:
        with self.ledger.key_lock("case", case_id):
            records = self.ledger.records()
            claim_view, state, feature = self._latest_projection(records, case_id, claim_code)
            snapshot = self._snapshot(records, claim_view, state, feature)
            created_at = datetime.now(UTC)
            run_id = f"analysis-run-{uuid.uuid4().hex}"
            try:
                output = NonClinicalReasoningOutput.model_validate(
                    reasoner.reason(claim_view)
                ).model_dump(mode="json")
            except ValidationError as exc:
                raise ValueError(
                    "reasoner output violates the synthetic non-clinical contract"
                ) from exc
            expected_output = expected_non_clinical_output(claim_view)
            if canonical_json_bytes(output) != canonical_json_bytes(expected_output):
                raise ValueError(
                    "reasoner output violates the exact synthetic non-clinical projection"
                )
            run = AnalysisRun(
                analysis_run_id=run_id,
                case_id=case_id,
                claim_view_id=claim_view.claim_view_id,
                snapshot_id=snapshot.snapshot_id,
                reasoner_name=reasoner.name,
                reasoner_version=reasoner.version,
                created_at=created_at,
                output=output,
                labels=("SYNTHETIC", "NON-CLINICAL"),
                source_locators=claim_view.source_locators,
                input_hash=snapshot.dependency_hash,
                component_version=COMPONENT_VERSION,
                policy_version=POLICY_VERSION,
                dependency_ids=(snapshot.snapshot_id, claim_view.claim_view_id),
            )
            self.ledger.append_once(
                "ANALYSIS_RUN_COMPLETED",
                run.analysis_run_id,
                {"analysis_run": run.model_dump(mode="json")},
            )
            return run

    @staticmethod
    def _latest_projection(records, case_id: str, claim_code: str):
        states = {
            record["payload"]["state"]["state_id"]: AnkleStateVersion.model_validate(
                record["payload"]["state"]
            )
            for record in records
            if record["kind"] == "ANKLE_STATE_MATERIALIZED"
            and record["payload"]["state"]["case_id"] == case_id
        }
        claim_views = [
            ClaimView.model_validate(record["payload"]["claim_view"])
            for record in records
            if record["kind"] == "CLAIM_VIEW_MATERIALIZED"
            and record["payload"]["claim_view"]["case_id"] == case_id
            and record["payload"]["claim_view"]["claim_code"] == claim_code
        ]
        if not claim_views:
            raise ValueError(f"no claim view exists for {case_id} / {claim_code}")
        latest_state_version = max(states[view.state_id].version for view in claim_views)
        latest_state_ids = {
            view.state_id
            for view in claim_views
            if states[view.state_id].version == latest_state_version
        }
        claim_view = next(
            view for view in reversed(claim_views) if view.state_id in latest_state_ids
        )
        state = states[claim_view.state_id]
        features = [
            FeatureVersion.model_validate(record["payload"]["feature"])
            for record in records
            if record["kind"] == "FEATURE_MATERIALIZED"
            and record["payload"]["feature"]["case_id"] == case_id
            and record["payload"]["feature"]["dependency_ids"][0] == state.state_id
        ]
        if not features:
            raise ValueError("claim state has no materialized feature")
        feature = features[-1]
        return claim_view, state, feature

    def _snapshot(
        self,
        records,
        claim_view: ClaimView,
        state: AnkleStateVersion,
        feature: FeatureVersion,
    ) -> SnapshotManifest:
        raw_ids = tuple(dict.fromkeys(locator.artifact_id for locator in claim_view.source_locators))
        selected_event_ids = set(state.event_ids)
        event_versions = {
            record["payload"]["event"]["event_id"]: (
                record["payload"]["event"]["component_version"],
                record["payload"]["event"]["policy_version"],
                record["payload"]["event"]["verification_decision_id"],
            )
            for record in records
            if record["kind"] == "CLINICAL_EVENT_PROMOTED"
            and record["payload"]["event"]["event_id"] in selected_event_ids
        }
        selected_decision_ids = {item[2] for item in event_versions.values()}
        decision_versions = {
            record["payload"]["decision"]["decision_id"]: (
                record["payload"]["decision"]["component_version"],
                record["payload"]["decision"]["policy_version"],
            )
            for record in records
            if record["kind"] == "VERIFICATION_DECIDED"
            and record["payload"]["decision"]["decision_id"] in selected_decision_ids
        }
        dependency_payload = {
            "case_id": claim_view.case_id,
            "raw_ids": raw_ids,
            "event_ids": state.event_ids,
            "state_id": state.state_id,
            "feature_id": feature.feature_id,
            "claim_view_id": claim_view.claim_view_id,
            "pipeline_versions": {
                "state": state.component_version,
                "state_policy": state.policy_version,
                "feature": feature.component_version,
                "feature_policy": feature.policy_version,
                "claim": claim_view.component_version,
                "claim_policy": claim_view.policy_version,
            },
            "event_versions": event_versions,
            "verification_versions": decision_versions,
            "snapshot_component_version": COMPONENT_VERSION,
            "snapshot_policy_version": POLICY_VERSION,
        }
        dependency_hash = sha256_digest(canonical_json_bytes(dependency_payload))
        snapshot_id = deterministic_id("snapshot", dependency_payload)
        existing = next(
            (
                SnapshotManifest.model_validate(record["payload"]["snapshot"])
                for record in records
                if record["kind"] == "SNAPSHOT_SEALED"
                and record["payload"]["snapshot"]["snapshot_id"] == snapshot_id
            ),
            None,
        )
        if existing is not None:
            return existing
        snapshot = SnapshotManifest(
            snapshot_id=snapshot_id,
            case_id=claim_view.case_id,
            raw_artifact_ids=raw_ids,
            event_ids=state.event_ids,
            state_ids=(state.state_id,),
            feature_ids=(feature.feature_id,),
            claim_view_ids=(claim_view.claim_view_id,),
            pipeline_versions=dependency_payload["pipeline_versions"],
            dependency_hash=dependency_hash,
            created_at=datetime.now(UTC),
            source_locators=claim_view.source_locators,
            input_hash=dependency_hash,
            component_version=COMPONENT_VERSION,
            policy_version=POLICY_VERSION,
            dependency_ids=(state.state_id, feature.feature_id, claim_view.claim_view_id),
        )
        self.ledger.append_once(
            "SNAPSHOT_SEALED",
            snapshot.snapshot_id,
            {"snapshot": snapshot.model_dump(mode="json")},
        )
        return snapshot
