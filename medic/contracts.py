from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]


class FrozenDict(dict[str, Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("versioned contract payloads are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class FrozenList(list[Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("versioned contract payloads are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable


def deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, FrozenList):
        return value
    if isinstance(value, list):
        return FrozenList(deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(deep_freeze(item) for item in value)
    return value


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def model_post_init(self, __context: Any) -> None:
        del __context
        for field_name in type(self).model_fields:
            current = getattr(self, field_name)
            frozen = deep_freeze(current)
            if frozen is not current:
                object.__setattr__(self, field_name, frozen)


class IntegrityStatus(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    MISMATCH = "MISMATCH"


class VerificationStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"
    UNKNOWN = "UNKNOWN"


class ClaimReadiness(StrEnum):
    READY = "READY"
    READY_WITH_LIMITS = "READY_WITH_LIMITS"
    NOT_READY = "NOT_READY"


class RawArtifact(FrozenContract):
    artifact_id: NonEmptyStr
    case_id: NonEmptyStr
    content_hash: Sha256
    media_type: NonEmptyStr
    source_name: NonEmptyStr
    ingested_at: AwareDatetime


class SourceLocator(FrozenContract):
    artifact_id: NonEmptyStr
    locator_type: NonEmptyStr
    locator: NonEmptyStr


class DerivedContract(FrozenContract):
    source_locators: tuple[SourceLocator, ...] = Field(min_length=1)
    input_hash: Sha256
    component_version: NonEmptyStr
    policy_version: NonEmptyStr
    dependency_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)


class NormalizedSourceBlock(DerivedContract):
    block_id: NonEmptyStr
    artifact_id: NonEmptyStr
    block_type: NonEmptyStr
    raw_payload: dict[str, JsonValue]


class IntegrityAssessment(DerivedContract):
    assessment_id: NonEmptyStr
    artifact_id: NonEmptyStr
    status: IntegrityStatus
    reasons: tuple[str, ...] = ()


class CandidateFact(DerivedContract):
    candidate_id: NonEmptyStr
    case_id: NonEmptyStr
    semantic_type: NonEmptyStr
    field_name: NonEmptyStr
    side: NonEmptyStr
    episode_id: NonEmptyStr
    epoch_id: NonEmptyStr
    raw_value: JsonValue
    normalized_value: JsonValue | None = None
    high_impact: bool
    extraction_method: NonEmptyStr


class VerificationDecision(DerivedContract):
    decision_id: NonEmptyStr
    candidate_id: NonEmptyStr
    status: VerificationStatus
    decided_at: AwareDatetime
    reviewer: NonEmptyStr
    corrected_value: JsonValue | None = None
    corrected_effective_at: AwareDatetime | None = None
    corrected_side: NonEmptyStr | None = None
    corrected_episode_id: NonEmptyStr | None = None
    corrected_epoch_id: NonEmptyStr | None = None
    replaces_decision_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_corrected_value(self) -> VerificationDecision:
        if self.status is VerificationStatus.CORRECTED and self.corrected_value is None:
            raise ValueError("CORRECTED verification requires corrected_value")
        if self.status is not VerificationStatus.CORRECTED and self.corrected_value is not None:
            raise ValueError("corrected_value is only valid for CORRECTED verification")
        if self.status is not VerificationStatus.CORRECTED and self.corrected_effective_at is not None:
            raise ValueError("corrected_effective_at is only valid for CORRECTED verification")
        routing_corrections = (
            self.corrected_side,
            self.corrected_episode_id,
            self.corrected_epoch_id,
        )
        if self.status is not VerificationStatus.CORRECTED and any(routing_corrections):
            raise ValueError("routing corrections are only valid for CORRECTED verification")
        return self


class ClinicalEventVersion(DerivedContract):
    event_id: NonEmptyStr
    case_id: NonEmptyStr
    candidate_id: NonEmptyStr
    version: StrictPositiveInt
    semantic_type: NonEmptyStr
    field_name: NonEmptyStr
    side: NonEmptyStr
    episode_id: NonEmptyStr
    epoch_id: NonEmptyStr
    effective_at: AwareDatetime | None = None
    value: JsonValue
    verification_decision_id: NonEmptyStr
    supersedes_event_id: NonEmptyStr | None = None


class AnkleStateVersion(DerivedContract):
    state_id: NonEmptyStr
    case_id: NonEmptyStr
    version: StrictPositiveInt
    side: NonEmptyStr
    episode_id: NonEmptyStr
    epoch_id: NonEmptyStr
    event_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    fixation_construct: dict[str, JsonValue] = Field(default_factory=dict)
    healing_units: dict[str, JsonValue] = Field(default_factory=dict)
    symptoms_function_load: dict[str, JsonValue] = Field(default_factory=dict)
    dependency_hash: Sha256


class FeatureValue(FrozenContract):
    feature_id: NonEmptyStr
    definition_id: NonEmptyStr
    definition_version: NonEmptyStr
    input_fact_or_roi_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    source_locators: tuple[SourceLocator, ...] = Field(min_length=1)
    observed_or_derived: NonEmptyStr
    value: JsonValue
    unit: NonEmptyStr | None = None
    effective_start: AwareDatetime | None = None
    effective_end: AwareDatetime | None = None
    method_or_algorithm_version: NonEmptyStr
    qc_status: NonEmptyStr
    uncertainty: dict[str, JsonValue] | None = None
    repeatability: dict[str, JsonValue] | None = None
    missingness: NonEmptyStr | None = None
    allowed_uses: tuple[NonEmptyStr, ...] = Field(min_length=1)
    dependency_hash: Sha256


class FeatureVersion(DerivedContract):
    feature_id: NonEmptyStr
    case_id: NonEmptyStr
    version: StrictPositiveInt
    name: NonEmptyStr
    payload: FeatureValue
    dependency_hash: Sha256

    @model_validator(mode="after")
    def validate_payload_identity(self) -> FeatureVersion:
        if self.payload.feature_id != self.feature_id:
            raise ValueError("FeatureValue.feature_id must match FeatureVersion.feature_id")
        if self.payload.dependency_hash != self.dependency_hash:
            raise ValueError("FeatureValue.dependency_hash must match FeatureVersion.dependency_hash")
        return self


class ClaimView(DerivedContract):
    claim_view_id: NonEmptyStr
    case_id: NonEmptyStr
    claim_code: NonEmptyStr
    state_id: NonEmptyStr
    readiness: ClaimReadiness
    eligible_event_ids: tuple[NonEmptyStr, ...]
    supporting_event_ids: tuple[NonEmptyStr, ...] = ()
    contradicting_event_ids: tuple[NonEmptyStr, ...] = ()
    alternative_explanations: tuple[NonEmptyStr, ...] = ()
    conflict_ids: tuple[NonEmptyStr, ...] = ()
    missing_requirements: tuple[NonEmptyStr, ...] = ()
    uncertainty_notes: tuple[NonEmptyStr, ...] = ()
    dependency_hash: Sha256


class SnapshotManifest(DerivedContract):
    snapshot_id: NonEmptyStr
    case_id: NonEmptyStr
    raw_artifact_ids: tuple[NonEmptyStr, ...]
    event_ids: tuple[NonEmptyStr, ...]
    state_ids: tuple[NonEmptyStr, ...]
    feature_ids: tuple[NonEmptyStr, ...]
    claim_view_ids: tuple[NonEmptyStr, ...]
    pipeline_versions: dict[NonEmptyStr, NonEmptyStr]
    dependency_hash: Sha256
    created_at: AwareDatetime


class AnalysisRun(DerivedContract):
    analysis_run_id: NonEmptyStr
    case_id: NonEmptyStr
    claim_view_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    reasoner_name: NonEmptyStr
    reasoner_version: NonEmptyStr
    created_at: AwareDatetime
    output: dict[str, JsonValue]
    labels: tuple[NonEmptyStr, ...] = ("SYNTHETIC", "NON-CLINICAL")
