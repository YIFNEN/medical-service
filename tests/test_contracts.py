from __future__ import annotations

import importlib
import math
import unittest
import warnings
from datetime import UTC, datetime

from pydantic import ValidationError


class ContractTests(unittest.TestCase):
    def contracts(self):
        try:
            return importlib.import_module("medic.contracts")
        except ModuleNotFoundError as exc:
            self.fail(f"medic.contracts must exist: {exc}")

    def source_locator(self):
        contracts = self.contracts()
        return contracts.SourceLocator(
            artifact_id="artifact-001",
            locator_type="JSON_POINTER",
            locator="/facts/0",
        )

    def provenance(self):
        return {
            "source_locators": (self.source_locator(),),
            "input_hash": "sha256:" + "f" * 64,
            "component_version": "test-component/0.1.0",
            "policy_version": "test-policy/0.1.0",
            "dependency_ids": ("artifact-001",),
        }

    def test_contracts_are_frozen_and_forbid_unknown_fields(self):
        contracts = self.contracts()
        artifact = contracts.RawArtifact(
            artifact_id="artifact-001",
            case_id="synthetic-ankle-001",
            content_hash="sha256:" + "a" * 64,
            media_type="application/json",
            source_name="index_source.json",
            ingested_at=datetime(2026, 8, 18, tzinfo=UTC),
        )

        with self.assertRaises(ValidationError):
            contracts.RawArtifact(**(artifact.model_dump() | {"unexpected": True}))
        with self.assertRaises(ValidationError):
            artifact.source_name = "changed.json"

    def test_corrected_verification_requires_a_corrected_value(self):
        contracts = self.contracts()
        with self.assertRaises(ValidationError):
            contracts.VerificationDecision(
                decision_id="verification-001",
                candidate_id="candidate-001",
                status="CORRECTED",
                decided_at=datetime(2026, 8, 18, tzinfo=UTC),
                reviewer="USER",
                **self.provenance(),
            )

    def test_unknown_verification_cannot_smuggle_a_corrected_value(self):
        contracts = self.contracts()
        with self.assertRaises(ValidationError):
            contracts.VerificationDecision(
                decision_id="verification-001",
                candidate_id="candidate-001",
                status="UNKNOWN",
                corrected_value={"value": "right"},
                decided_at=datetime(2026, 8, 18, tzinfo=UTC),
                reviewer="USER",
                **self.provenance(),
            )

    def test_promoted_event_requires_verification_and_source_provenance(self):
        contracts = self.contracts()
        event = contracts.ClinicalEventVersion(
            event_id="event-001-v1",
            case_id="synthetic-ankle-001",
            candidate_id="candidate-001",
            version=1,
            semantic_type="PROCEDURE",
            field_name="index_orif",
            side="RIGHT",
            episode_id="episode-001",
            epoch_id="epoch-001",
            effective_at=datetime(2025, 1, 2, tzinfo=UTC),
            value={"procedure": "ORIF"},
            source_locators=(self.source_locator(),),
            verification_decision_id="verification-001",
            input_hash="sha256:" + "b" * 64,
            component_version="clinical-compiler/0.1.0",
            policy_version="clinical-event-policy/0.1.0",
            dependency_ids=("artifact-001", "candidate-001"),
        )

        self.assertEqual(event.verification_decision_id, "verification-001")
        self.assertEqual(event.source_locators[0].locator, "/facts/0")
        self.assertEqual(event.dependency_ids, ("artifact-001", "candidate-001"))

    def test_all_derived_contracts_require_a_provenance_envelope(self):
        contracts = self.contracts()
        derived_models = (
            contracts.NormalizedSourceBlock,
            contracts.IntegrityAssessment,
            contracts.CandidateFact,
            contracts.VerificationDecision,
            contracts.ClinicalEventVersion,
            contracts.AnkleStateVersion,
            contracts.FeatureVersion,
            contracts.ClaimView,
            contracts.SnapshotManifest,
            contracts.AnalysisRun,
        )
        envelope_fields = {
            "source_locators",
            "input_hash",
            "component_version",
            "policy_version",
            "dependency_ids",
        }

        for model in derived_models:
            with self.subTest(model=model.__name__):
                self.assertTrue(envelope_fields.issubset(model.model_fields))
                for field_name in envelope_fields:
                    self.assertTrue(model.model_fields[field_name].is_required())

    def test_provenance_envelope_rejects_empty_sources_and_dependencies(self):
        contracts = self.contracts()
        with self.assertRaises(ValidationError):
            contracts.IntegrityAssessment(
                assessment_id="integrity-001",
                artifact_id="artifact-001",
                status="PASS",
                reasons=(),
                source_locators=(),
                input_hash="sha256:" + "c" * 64,
                component_version="integrity-gate/0.1.0",
                policy_version="single-user-subject/0.1.0",
                dependency_ids=(),
            )

        with self.assertRaises(ValidationError):
            contracts.IntegrityAssessment(
                assessment_id="integrity-001",
                artifact_id="artifact-001",
                status="PASS",
                reasons=(),
                source_locators=(self.source_locator(),),
                input_hash="sha256:" + "c" * 64,
                component_version="integrity-gate/0.1.0",
                policy_version="single-user-subject/0.1.0",
                dependency_ids=(),
            )

    def test_provenance_envelope_rejects_blank_values_and_invalid_hash(self):
        contracts = self.contracts()
        invalid_envelopes = (
            self.provenance() | {"input_hash": ""},
            self.provenance() | {"input_hash": "sha256:not-a-hash"},
            self.provenance() | {"component_version": " "},
            self.provenance() | {"policy_version": ""},
            self.provenance() | {"dependency_ids": ("",)},
        )
        for envelope in invalid_envelopes:
            with self.subTest(envelope=envelope), self.assertRaises(ValidationError):
                contracts.IntegrityAssessment(
                    assessment_id="integrity-001",
                    artifact_id="artifact-001",
                    status="PASS",
                    reasons=(),
                    **envelope,
                )

        with self.assertRaises(ValidationError):
            contracts.SourceLocator(artifact_id="", locator_type="JSON_POINTER", locator="/facts/0")

    def test_feature_version_uses_a_typed_feature_value(self):
        contracts = self.contracts()
        payload = contracts.FeatureValue(
            feature_id="feature-001-v1",
            definition_id="elapsed-time-since-orif",
            definition_version="1.0.0",
            input_fact_or_roi_ids=("event-001-v1",),
            source_locators=(self.source_locator(),),
            observed_or_derived="DERIVED",
            value=30,
            unit="d",
            effective_start=datetime(2025, 2, 1, tzinfo=UTC),
            effective_end=None,
            method_or_algorithm_version="elapsed-time/1.0.0",
            qc_status="VALIDATED",
            uncertainty={"lower": 29, "upper": 31},
            repeatability=None,
            missingness=None,
            allowed_uses=("ANKLE_POST_ORIF_STRUCTURAL_STATE",),
            dependency_hash="sha256:" + "d" * 64,
        )
        feature = contracts.FeatureVersion(
            feature_id="feature-001-v1",
            case_id="synthetic-ankle-001",
            version=1,
            name="days_since_orif",
            payload=payload,
            source_locators=(self.source_locator(),),
            input_hash="sha256:" + "d" * 64,
            component_version="feature-builder/0.1.0",
            policy_version="ankle-feature-policy/0.1.0",
            dependency_ids=("event-001-v1",),
            dependency_hash="sha256:" + "d" * 64,
        )

        self.assertIsInstance(feature.payload, contracts.FeatureValue)
        self.assertEqual(feature.payload.allowed_uses, ("ANKLE_POST_ORIF_STRUCTURAL_STATE",))

    def test_feature_payload_is_validated_from_mapping_and_id_must_match(self):
        contracts = self.contracts()
        feature_data = {
            "feature_id": "feature-other-v1",
            "definition_id": "elapsed-time-since-orif",
            "definition_version": "1.0.0",
            "input_fact_or_roi_ids": ("event-001-v1",),
            "source_locators": (self.source_locator().model_dump(),),
            "observed_or_derived": "DERIVED",
            "value": {"days": 30},
            "unit": "d",
            "effective_start": datetime(2025, 2, 1, tzinfo=UTC),
            "effective_end": None,
            "method_or_algorithm_version": "elapsed-time/1.0.0",
            "qc_status": "VALIDATED",
            "uncertainty": None,
            "repeatability": None,
            "missingness": None,
            "allowed_uses": ("ANKLE_POST_ORIF_STRUCTURAL_STATE",),
            "dependency_hash": "sha256:" + "d" * 64,
        }
        with self.assertRaises(ValidationError):
            contracts.FeatureVersion(
                feature_id="feature-001-v1",
                case_id="synthetic-ankle-001",
                version=1,
                name="days_since_orif",
                payload=feature_data,
                source_locators=(self.source_locator(),),
                input_hash="sha256:" + "d" * 64,
                component_version="feature-builder/0.1.0",
                policy_version="ankle-feature-policy/0.1.0",
                dependency_ids=("event-001-v1",),
                dependency_hash="sha256:" + "d" * 64,
            )

        malformed = feature_data | {"feature_id": "feature-001-v1", "allowed_uses": ()}
        with self.assertRaises(ValidationError):
            contracts.FeatureVersion(
                feature_id="feature-001-v1",
                case_id="synthetic-ankle-001",
                version=1,
                name="days_since_orif",
                payload=malformed,
                source_locators=(self.source_locator(),),
                input_hash="sha256:" + "d" * 64,
                component_version="feature-builder/0.1.0",
                policy_version="ankle-feature-policy/0.1.0",
                dependency_ids=("event-001-v1",),
                dependency_hash="sha256:" + "d" * 64,
            )

    def test_nested_payloads_are_deeply_immutable(self):
        contracts = self.contracts()
        artifact = contracts.RawArtifact(
            artifact_id="artifact-001",
            case_id="synthetic-ankle-001",
            content_hash="sha256:" + "a" * 64,
            media_type="application/json",
            source_name="index_source.json",
            ingested_at=datetime(2026, 8, 18, tzinfo=UTC),
        )
        block = contracts.NormalizedSourceBlock(
            block_id="block-001",
            artifact_id=artifact.artifact_id,
            block_type="FACT_LIST",
            raw_payload={"facts": [{"value": "ORIF"}]},
            **self.provenance(),
        )

        with self.assertRaises(TypeError):
            block.raw_payload["facts"] = ()
        with self.assertRaises(TypeError):
            block.raw_payload["facts"][0]["value"] = "changed"

        original = {"facts": [{"value": "ORIF"}]}
        copied = contracts.NormalizedSourceBlock(
            block_id="block-002",
            artifact_id=artifact.artifact_id,
            block_type="FACT_LIST",
            raw_payload=original,
            **self.provenance(),
        )
        original["facts"][0]["value"] = "mutated outside"
        self.assertEqual(copied.raw_payload["facts"][0]["value"], "ORIF")

        with self.assertRaises(ValidationError):
            contracts.CandidateFact(
                candidate_id="candidate-001",
                case_id="synthetic-ankle-001",
                semantic_type="OBSERVATION",
                field_name="unsafe_payload",
                side="RIGHT",
                episode_id="episode-001",
                epoch_id="epoch-001",
                raw_value=bytearray(b"mutable"),
                normalized_value=None,
                high_impact=False,
                extraction_method="DETERMINISTIC",
                **self.provenance(),
            )

    def test_non_finite_numbers_are_rejected_before_hash_or_serialization(self):
        contracts = self.contracts()
        for invalid_value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=invalid_value), self.assertRaises(ValidationError):
                contracts.CandidateFact(
                    candidate_id="candidate-001",
                    case_id="synthetic-ankle-001",
                    semantic_type="OBSERVATION",
                    field_name="numeric_value",
                    side="RIGHT",
                    episode_id="episode-001",
                    epoch_id="epoch-001",
                    raw_value=invalid_value,
                    normalized_value=None,
                    high_impact=True,
                    extraction_method="DETERMINISTIC",
                    **self.provenance(),
                )

    def test_immutable_json_payload_round_trips_without_serializer_warnings(self):
        contracts = self.contracts()
        block = contracts.NormalizedSourceBlock(
            block_id="block-001",
            artifact_id="artifact-001",
            block_type="FACT_LIST",
            raw_payload={"facts": [{"value": "ORIF"}]},
            **self.provenance(),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            python_dump = block.model_dump()
            json_dump = block.model_dump_json()

        self.assertEqual(caught, [])
        self.assertIsInstance(python_dump["raw_payload"]["facts"], list)
        restored = contracts.NormalizedSourceBlock.model_validate_json(json_dump)
        self.assertEqual(restored.model_dump(), python_dump)

    def test_stable_ids_and_dependency_hashes_reject_blanks(self):
        contracts = self.contracts()
        with self.assertRaises(ValidationError):
            contracts.ClinicalEventVersion(
                event_id="",
                case_id="synthetic-ankle-001",
                candidate_id="candidate-001",
                version=1,
                semantic_type="PROCEDURE",
                field_name="index_orif",
                side="RIGHT",
                episode_id="episode-001",
                epoch_id="epoch-001",
                effective_at=datetime(2025, 1, 2, tzinfo=UTC),
                value={"procedure": "ORIF"},
                verification_decision_id="verification-001",
                **self.provenance(),
            )

        with self.assertRaises(ValidationError):
            contracts.AnkleStateVersion(
                state_id="state-001-v1",
                case_id="synthetic-ankle-001",
                version=1,
                side="RIGHT",
                episode_id="episode-001",
                epoch_id="epoch-001",
                event_ids=("event-001-v1",),
                dependency_hash="",
                **self.provenance(),
            )

    def test_feature_version_and_payload_dependency_hashes_must_match(self):
        contracts = self.contracts()
        payload = {
            "feature_id": "feature-001-v1",
            "definition_id": "elapsed-time-since-orif",
            "definition_version": "1.0.0",
            "input_fact_or_roi_ids": ("event-001-v1",),
            "source_locators": (self.source_locator().model_dump(),),
            "observed_or_derived": "DERIVED",
            "value": 30,
            "unit": "d",
            "effective_start": datetime(2025, 2, 1, tzinfo=UTC),
            "effective_end": None,
            "method_or_algorithm_version": "elapsed-time/1.0.0",
            "qc_status": "VALIDATED",
            "uncertainty": None,
            "repeatability": None,
            "missingness": None,
            "allowed_uses": ("ANKLE_POST_ORIF_STRUCTURAL_STATE",),
            "dependency_hash": "sha256:" + "b" * 64,
        }
        with self.assertRaises(ValidationError):
            contracts.FeatureVersion(
                feature_id="feature-001-v1",
                case_id="synthetic-ankle-001",
                version=1,
                name="days_since_orif",
                payload=payload,
                source_locators=(self.source_locator(),),
                input_hash="sha256:" + "d" * 64,
                component_version="feature-builder/0.1.0",
                policy_version="ankle-feature-policy/0.1.0",
                dependency_ids=("event-001-v1",),
                dependency_hash="sha256:" + "c" * 64,
            )

    def test_versions_are_strict_integers_and_audit_times_are_timezone_aware(self):
        contracts = self.contracts()
        common = {
            "event_id": "event-001-v1",
            "case_id": "synthetic-ankle-001",
            "candidate_id": "candidate-001",
            "semantic_type": "PROCEDURE",
            "field_name": "index_orif",
            "side": "RIGHT",
            "episode_id": "episode-001",
            "epoch_id": "epoch-001",
            "value": {"procedure": "ORIF"},
            "verification_decision_id": "verification-001",
            **self.provenance(),
        }
        with self.assertRaises(ValidationError):
            contracts.ClinicalEventVersion(
                **common,
                version=True,
                effective_at=datetime(2025, 1, 2, tzinfo=UTC),
            )
        with self.assertRaises(ValidationError):
            contracts.ClinicalEventVersion(
                **common,
                version=1,
                effective_at=datetime(2025, 1, 2),
            )

    def test_claim_view_exposes_eligible_facts_alternatives_and_conflicts(self):
        contracts = self.contracts()
        fields = contracts.ClaimView.model_fields
        self.assertIn("eligible_event_ids", fields)
        self.assertIn("alternative_explanations", fields)
        self.assertIn("conflict_ids", fields)

    def test_required_vertical_slice_models_are_available(self):
        contracts = self.contracts()
        required = {
            "RawArtifact",
            "SourceLocator",
            "NormalizedSourceBlock",
            "IntegrityAssessment",
            "CandidateFact",
            "VerificationDecision",
            "ClinicalEventVersion",
            "AnkleStateVersion",
            "FeatureValue",
            "FeatureVersion",
            "ClaimView",
            "SnapshotManifest",
            "AnalysisRun",
        }
        self.assertTrue(required.issubset(set(dir(contracts))))


if __name__ == "__main__":
    unittest.main()
