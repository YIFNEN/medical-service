from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    JsonValue,
    StrictBool,
    ValidationError,
    model_validator,
)

from .contracts import (
    CandidateFact,
    IntegrityAssessment,
    IntegrityStatus,
    NormalizedSourceBlock,
    RawArtifact,
    SourceLocator,
    NonEmptyStr,
)
from .ledger import (
    AppendOnlyJsonlLedger,
    ContentAddressedStore,
    canonical_json_bytes,
    deterministic_id,
    sha256_digest,
)


COMPONENT_VERSION = "clinical-json-compiler/0.1.0"
POLICY_VERSION = "single-user-integrity/0.1.0"


class SourceSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_alias: NonEmptyStr


class SourceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    semantic_type: NonEmptyStr
    field_name: NonEmptyStr
    side: NonEmptyStr
    episode_id: NonEmptyStr
    epoch_id: NonEmptyStr
    effective_at: AwareDatetime
    value: dict[str, JsonValue]
    high_impact: StrictBool

    @model_validator(mode="after")
    def validate_side_consistency(self) -> "SourceFact":
        value_side = self.value.get("side")
        if value_side is not None and value_side != self.side:
            raise ValueError("fact value.side must match the canonical top-level side")
        return self


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: NonEmptyStr
    subject: SourceSubject
    facts: tuple[SourceFact, ...]


@dataclass(frozen=True)
class CompileResult:
    artifact: RawArtifact
    compile_id: str
    reused: bool
    integrity: IntegrityAssessment
    blocks: tuple[NormalizedSourceBlock, ...]
    candidates: tuple[CandidateFact, ...]
    review_queue: tuple[CandidateFact, ...]


class ClinicalCompiler:
    def __init__(
        self,
        store: ContentAddressedStore,
        ledger: AppendOnlyJsonlLedger,
        expected_subject_alias: str,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.expected_subject_alias = expected_subject_alias

    def compile_json(self, case_id: str, source_path: Path) -> CompileResult:
        artifact = self.store.register_raw(case_id, source_path)
        compile_id = deterministic_id(
            "compile",
            {
                "artifact_hash": artifact.content_hash,
                "case_id": case_id,
                "component_version": COMPONENT_VERSION,
                "policy_version": POLICY_VERSION,
                "expected_subject_alias": self.expected_subject_alias,
            },
        )
        with self.store.key_lock("compile", compile_id):
            return self._compile_registered(case_id, artifact, compile_id)

    def load_compile(self, compile_id: str) -> CompileResult:
        completion = next(
            (
                record
                for record in self.ledger.records()
                if record["kind"] == "COMPILE_COMPLETED"
                and record["natural_key"] == compile_id
            ),
            None,
        )
        if completion is None:
            raise KeyError(f"unknown compile_id: {compile_id}")
        artifact_id = completion["payload"]["artifact_id"]
        metadata_path = self.store.metadata_root / f"{artifact_id}.json"
        artifact = RawArtifact.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        return self._restore(
            self.store.compiled_path(compile_id),
            expected_compile_id=compile_id,
            expected_artifact=artifact,
            expected_manifest_hash=completion["payload"]["manifest_hash"],
            reused=True,
        )

    def list_compiles(self, case_id: str) -> tuple[CompileResult, ...]:
        compile_ids = [
            record["payload"]["compile_id"]
            for record in self.ledger.records()
            if record["kind"] == "COMPILE_COMPLETED"
        ]
        return tuple(
            result
            for result in (self.load_compile(compile_id) for compile_id in compile_ids)
            if result.artifact.case_id == case_id
        )

    def _compile_registered(
        self,
        case_id: str,
        artifact: RawArtifact,
        compile_id: str,
    ) -> CompileResult:
        cached_path = self.store.compiled_path(compile_id)
        if cached_path.exists():
            completion = next(
                (
                    record
                    for record in self.ledger.records()
                    if record["kind"] == "COMPILE_COMPLETED"
                    and record["natural_key"] == compile_id
                ),
                None,
            )
            restored = self._restore(
                cached_path,
                expected_compile_id=compile_id,
                expected_artifact=artifact,
                expected_manifest_hash=(
                    completion["payload"].get("manifest_hash") if completion else None
                ),
                reused=True,
            )
            if completion is None:
                rebuilt = self._build_result(case_id, artifact, compile_id, reused=False)
                if self._dump(restored) != self._dump(rebuilt):
                    raise RuntimeError("compile cache integrity raw reconstruction mismatch")
            manifest_hash = sha256_digest(canonical_json_bytes(self._dump(restored)))
            self._record_completion(restored, manifest_hash)
            return restored

        result = self._build_result(case_id, artifact, compile_id, reused=False)
        manifest = self._dump(result)
        manifest_hash = sha256_digest(canonical_json_bytes(manifest))
        self.store.save_compiled(compile_id, manifest)
        self._record_completion(result, manifest_hash)
        return result

    def _build_result(
        self,
        case_id: str,
        artifact: RawArtifact,
        compile_id: str,
        reused: bool,
    ) -> CompileResult:
        raw_bytes = self.store.read_raw(artifact).read_bytes()
        try:
            source = SourceDocument.model_validate_json(raw_bytes)
        except ValidationError as exc:
            raise ValueError(f"source JSON schema validation failed: {exc}") from exc
        if source.case_id != case_id:
            raise ValueError("source case_id does not match the requested case")

        root_locator = SourceLocator(
            artifact_id=artifact.artifact_id,
            locator_type="JSON_POINTER",
            locator="/",
        )
        subject_alias = source.subject.local_alias
        status = (
            IntegrityStatus.PASS
            if subject_alias == self.expected_subject_alias
            else IntegrityStatus.MISMATCH
        )
        reasons = () if status is IntegrityStatus.PASS else ("SUBJECT_ALIAS_MISMATCH",)
        integrity_id = deterministic_id(
            "integrity",
            {
                "artifact_id": artifact.artifact_id,
                "expected_alias": self.expected_subject_alias,
                "observed_alias": subject_alias,
                "policy_version": POLICY_VERSION,
            },
        )
        integrity = IntegrityAssessment(
            assessment_id=integrity_id,
            artifact_id=artifact.artifact_id,
            status=status,
            reasons=reasons,
            source_locators=(root_locator,),
            input_hash=artifact.content_hash,
            component_version=COMPONENT_VERSION,
            policy_version=POLICY_VERSION,
            dependency_ids=(artifact.artifact_id,),
        )

        blocks: tuple[NormalizedSourceBlock, ...] = ()
        candidates: tuple[CandidateFact, ...] = ()
        if status is IntegrityStatus.PASS:
            compiled_facts = tuple(
                self._compile_fact(case_id, artifact, integrity, fact, index)
                for index, fact in enumerate(source.facts)
            )
            blocks = tuple(item[0] for item in compiled_facts)
            candidates = tuple(item[1] for item in compiled_facts)
        review_queue = tuple(candidate for candidate in candidates if candidate.high_impact)
        result = CompileResult(
            artifact=artifact,
            compile_id=compile_id,
            reused=reused,
            integrity=integrity,
            blocks=blocks,
            candidates=candidates,
            review_queue=review_queue,
        )
        return result

    def _record_completion(self, result: CompileResult, manifest_hash: str) -> None:
        self.ledger.append_once(
            kind="COMPILE_COMPLETED",
            natural_key=result.compile_id,
            payload={
                "artifact_id": result.artifact.artifact_id,
                "compile_id": result.compile_id,
                "manifest_hash": manifest_hash,
                "integrity_status": result.integrity.status.value,
                "candidate_ids": [candidate.candidate_id for candidate in result.candidates],
            },
        )

    def _compile_fact(
        self,
        case_id: str,
        artifact: RawArtifact,
        integrity: IntegrityAssessment,
        fact: SourceFact,
        index: int,
    ) -> tuple[NormalizedSourceBlock, CandidateFact]:
        locator = SourceLocator(
            artifact_id=artifact.artifact_id,
            locator_type="JSON_POINTER",
            locator=f"/facts/{index}",
        )
        fact_payload = fact.model_dump(mode="json")
        fact_hash = sha256_digest(canonical_json_bytes(fact_payload))
        block_id = deterministic_id(
            "block",
            {"artifact_id": artifact.artifact_id, "locator": locator.locator},
        )
        block = NormalizedSourceBlock(
            block_id=block_id,
            artifact_id=artifact.artifact_id,
            block_type="CLINICAL_FACT_JSON",
            raw_payload=fact_payload,
            source_locators=(locator,),
            input_hash=fact_hash,
            component_version=COMPONENT_VERSION,
            policy_version=POLICY_VERSION,
            dependency_ids=(artifact.artifact_id, integrity.assessment_id),
        )
        candidate_id = deterministic_id(
            "candidate",
            {
                "block_id": block.block_id,
                "semantic_type": fact.semantic_type,
                "field_name": fact.field_name,
                "side": fact.side,
                "episode_id": fact.episode_id,
                "epoch_id": fact.epoch_id,
                "value": fact.value,
            },
        )
        candidate = CandidateFact(
            candidate_id=candidate_id,
            case_id=case_id,
            semantic_type=fact.semantic_type,
            field_name=fact.field_name,
            side=fact.side,
            episode_id=fact.episode_id,
            epoch_id=fact.epoch_id,
            raw_value=fact.value,
            normalized_value=fact.value,
            high_impact=fact.high_impact,
            extraction_method="DETERMINISTIC_JSON",
            source_locators=(locator,),
            input_hash=fact_hash,
            component_version=COMPONENT_VERSION,
            policy_version=POLICY_VERSION,
            dependency_ids=(block.block_id, integrity.assessment_id),
        )
        return block, candidate

    @staticmethod
    def _dump(result: CompileResult) -> dict[str, Any]:
        return {
            "artifact": result.artifact.model_dump(mode="json"),
            "compile_id": result.compile_id,
            "integrity": result.integrity.model_dump(mode="json"),
            "blocks": [block.model_dump(mode="json") for block in result.blocks],
            "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
            "review_queue_ids": [candidate.candidate_id for candidate in result.review_queue],
        }

    @staticmethod
    def _restore(
        path: Path,
        expected_compile_id: str,
        expected_artifact: RawArtifact,
        expected_manifest_hash: str | None,
        reused: bool,
    ) -> CompileResult:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if set(wrapper) != {"manifest", "manifest_hash"}:
            raise RuntimeError("compile cache integrity envelope is invalid")
        payload = wrapper["manifest"]
        actual_hash = sha256_digest(canonical_json_bytes(payload))
        if actual_hash != wrapper["manifest_hash"]:
            raise RuntimeError("compile cache integrity hash mismatch")
        if expected_manifest_hash is not None and actual_hash != expected_manifest_hash:
            raise RuntimeError("compile cache integrity ledger mismatch")
        if payload.get("compile_id") != expected_compile_id:
            raise RuntimeError("compile cache integrity key mismatch")
        artifact = RawArtifact.model_validate(payload["artifact"])
        if artifact != expected_artifact:
            raise RuntimeError("compile cache integrity artifact mismatch")
        integrity = IntegrityAssessment.model_validate(payload["integrity"])
        if (
            integrity.artifact_id != artifact.artifact_id
            or integrity.input_hash != artifact.content_hash
            or integrity.component_version != COMPONENT_VERSION
            or integrity.policy_version != POLICY_VERSION
        ):
            raise RuntimeError("compile cache integrity provenance mismatch")
        blocks = tuple(NormalizedSourceBlock.model_validate(item) for item in payload["blocks"])
        candidates = tuple(CandidateFact.model_validate(item) for item in payload["candidates"])
        block_ids = {block.block_id for block in blocks}
        if any(candidate.dependency_ids[0] not in block_ids for candidate in candidates):
            raise RuntimeError("compile cache integrity dependency mismatch")
        queue_ids = set(payload["review_queue_ids"])
        canonical_queue_ids = [
            candidate.candidate_id for candidate in candidates if candidate.high_impact
        ]
        if payload["review_queue_ids"] != canonical_queue_ids:
            raise RuntimeError("compile cache integrity review queue mismatch")
        return CompileResult(
            artifact=artifact,
            compile_id=payload["compile_id"],
            reused=reused,
            integrity=integrity,
            blocks=blocks,
            candidates=candidates,
            review_queue=tuple(
                candidate for candidate in candidates if candidate.candidate_id in queue_ids
            ),
        )
