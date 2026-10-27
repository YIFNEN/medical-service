from __future__ import annotations

import importlib
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


FIXTURES = Path(__file__).parent / "fixtures" / "clinical_compiler"


class VerticalSliceTests(unittest.TestCase):
    def build_pipeline(self, root: Path):
        try:
            ledger_module = importlib.import_module("medic.ledger")
            compiler_module = importlib.import_module("medic.compiler")
            ankle_module = importlib.import_module("medic.ankle")
        except ModuleNotFoundError as exc:
            self.fail(f"vertical slice modules must exist: {exc}")
        store = ledger_module.ContentAddressedStore(root)
        ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")
        compiler = compiler_module.ClinicalCompiler(store, ledger, "LOCAL-SELF-001")
        projector = ankle_module.ClinicalAnkleProjector(ledger)
        return ledger, compiler, projector

    def test_review_and_materialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger, compiler, projector = self.build_pipeline(Path(temp_dir))
            compiled = compiler.compile_json(
                "synthetic-ankle-001", FIXTURES / "index_source.json"
            )

            result = projector.review_and_materialize(
                compiled, FIXTURES / "index_review.json"
            )

            self.assertEqual(len(result.events), 4)
            self.assertEqual(len(result.user_decisions), 3)
            self.assertEqual(len(result.system_decisions), 1)
            self.assertEqual(result.state.version, 1)
            self.assertEqual(result.state.side, "RIGHT")
            self.assertEqual(result.state.episode_id, "episode-index-fracture")
            self.assertEqual(result.state.epoch_id, "epoch-index-orif")
            self.assertEqual(result.state.fixation_construct["procedure"], "ORIF")
            self.assertEqual(result.feature.name, "ANKLE_POST_ORIF_STRUCTURAL_COVERAGE")
            self.assertEqual(result.claim_view.claim_code, "ANKLE_POST_ORIF_STRUCTURAL_STATE")
            self.assertEqual(result.claim_view.readiness, "READY")
            self.assertEqual(len(result.claim_view.eligible_event_ids), 4)
            self.assertEqual(result.claim_view.missing_requirements, ())
            for event in result.events:
                self.assertTrue(event.source_locators)
            promoted = [r for r in ledger.records() if r["kind"] == "CLINICAL_EVENT_PROMOTED"]
            self.assertEqual(len(promoted), 4)

    def test_incremental_followup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger, compiler, projector = self.build_pipeline(Path(temp_dir))
            index = compiler.compile_json(
                "synthetic-ankle-001", FIXTURES / "index_source.json"
            )
            first = projector.review_and_materialize(index, FIXTURES / "index_review.json")
            repeated = projector.review_and_materialize(index, FIXTURES / "index_review.json")
            self.assertEqual(first.events, repeated.events)

            followup = compiler.compile_json(
                "synthetic-ankle-001", FIXTURES / "followup_source.json"
            )
            self.assertEqual(len(followup.review_queue), 2)
            second = projector.review_and_materialize(
                followup, FIXTURES / "followup_review.json"
            )

            self.assertEqual(len(second.new_events), 2)
            self.assertEqual(second.state.version, 2)
            self.assertNotEqual(second.state.state_id, first.state.state_id)
            self.assertIn(first.state.state_id, second.state.dependency_ids)
            self.assertEqual(
                second.state.symptoms_function_load["ankle_pain"]["severity_0_10"], 2
            )
            self.assertEqual(len(second.claim_view.eligible_event_ids), 6)
            decisions = [r for r in ledger.records() if r["kind"] == "VERIFICATION_DECIDED"]
            self.assertEqual(len(decisions), 6)
            promoted = [r for r in ledger.records() if r["kind"] == "CLINICAL_EVENT_PROMOTED"]
            self.assertEqual(len(promoted), 6)

    def test_verify_once_ignores_changed_review_audit_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            later_payload = json.loads((FIXTURES / "index_review.json").read_text(encoding="utf-8"))
            later_payload["decided_at"] = "2026-08-19T12:00:00+09:00"
            later = root / "later-review.json"
            later.write_text(json.dumps(later_payload), encoding="utf-8")

            repeated = projector.review_and_materialize(compiled, later)

            self.assertEqual(repeated.state.state_id, first.state.state_id)
            self.assertEqual(repeated.new_events, ())
            self.assertEqual(
                len([r for r in ledger.records() if r["kind"] == "VERIFICATION_DECIDED"]), 4
            )

    def test_unknown_revision_retires_prior_event_from_claim_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            injury = next(
                decision
                for decision in first.user_decisions
                if decision.candidate_id == compiled.candidates[0].candidate_id
            )
            unknown_path = root / "unknown.json"
            unknown_path.write_text(
                json.dumps(
                    {
                        "reviewer": "LOCAL_USER",
                        "decided_at": "2026-08-19T12:00:00+09:00",
                        "decisions": [
                            {
                                "candidate_id": injury.candidate_id,
                                "status": "UNKNOWN",
                                "replaces_decision_id": injury.decision_id,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            revised = projector.review_and_materialize(compiled, unknown_path)

            self.assertEqual(revised.claim_view.readiness, "NOT_READY")
            self.assertIn("index_injury", revised.claim_view.missing_requirements)
            self.assertEqual(len(revised.claim_view.eligible_event_ids), 3)

    def test_claim_readiness_requires_matching_side_episode_and_epoch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            payload = json.loads((FIXTURES / "index_source.json").read_text(encoding="utf-8"))
            payload["facts"][1]["side"] = "LEFT"
            payload["facts"][1]["value"]["side"] = "LEFT"
            mixed = root / "mixed.json"
            mixed.write_text(json.dumps(payload), encoding="utf-8")
            compiled = compiler.compile_json("synthetic-ankle-001", mixed)
            review = root / "mixed-review.json"
            review.write_text(
                json.dumps(
                    {
                        "reviewer": "LOCAL_USER",
                        "decided_at": "2026-08-18T12:00:00+09:00",
                        "decisions": [
                            {"candidate_id": candidate.candidate_id, "status": "CONFIRMED"}
                            for candidate in compiled.review_queue
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = projector.review_and_materialize(compiled, review)

            self.assertEqual(result.claim_view.readiness, "NOT_READY")
            self.assertIn("index_orif", result.claim_view.missing_requirements)

    def test_component_version_change_creates_new_materialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            ankle_module = importlib.import_module("medic.ankle")

            with patch.object(
                ankle_module, "STATE_COMPONENT_VERSION", "ankle-state-materializer/0.2.0"
            ):
                second = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")

            self.assertEqual(second.state.version, 2)
            self.assertNotEqual(second.state.state_id, first.state.state_id)
            self.assertEqual(second.state.component_version, "ankle-state-materializer/0.2.0")
            self.assertEqual(second.new_events, ())
            self.assertEqual(len(second.user_decisions), 3)

    def test_corrected_side_requires_and_uses_structured_routing_correction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            injury = first.user_decisions[0]
            invalid = root / "invalid-side-review.json"
            base = {
                "reviewer": "LOCAL_USER",
                "decided_at": "2026-08-19T12:00:00+09:00",
                "decisions": [
                    {
                        "candidate_id": injury.candidate_id,
                        "status": "CORRECTED",
                        "corrected_value": {
                            "side": "LEFT",
                            "pattern": "TRIMALLEOLAR_FRACTURE_DISLOCATION",
                        },
                        "replaces_decision_id": injury.decision_id,
                    }
                ],
            }
            invalid.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrected_side"):
                projector.review_and_materialize(compiled, invalid)

            base["decisions"][0]["corrected_side"] = "LEFT"
            valid = root / "valid-side-review.json"
            valid.write_text(json.dumps(base), encoding="utf-8")
            revised = projector.review_and_materialize(compiled, valid)

            corrected_event = next(
                event for event in revised.events if event.candidate_id == injury.candidate_id
            )
            self.assertEqual(corrected_event.side, "LEFT")
            self.assertEqual(corrected_event.value["side"], "LEFT")
            self.assertEqual(revised.claim_view.readiness, "NOT_READY")

    def test_reverse_corrected_side_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            injury = first.user_decisions[0]
            review = root / "reverse-side.json"
            review.write_text(
                json.dumps(
                    {
                        "reviewer": "LOCAL_USER",
                        "decided_at": "2026-08-19T12:00:00+09:00",
                        "decisions": [
                            {
                                "candidate_id": injury.candidate_id,
                                "status": "CORRECTED",
                                "corrected_value": {
                                    "side": "RIGHT",
                                    "pattern": "TRIMALLEOLAR_FRACTURE_DISLOCATION",
                                },
                                "corrected_side": "LEFT",
                                "replaces_decision_id": injury.decision_id,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "resolved corrected_side"):
                projector.review_and_materialize(compiled, review)

    def test_event_version_change_reuses_review_but_reprojects_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            ankle_module = importlib.import_module("medic.ankle")

            with patch.object(
                ankle_module, "EVENT_COMPONENT_VERSION", "clinical-event-projector/0.2.0"
            ):
                second = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")

            self.assertEqual(len(second.new_events), 4)
            self.assertTrue(
                all(event.component_version == "clinical-event-projector/0.2.0" for event in second.events)
            )
            self.assertEqual(second.state.version, 2)
            self.assertEqual(
                len([record for record in ledger.records() if record["kind"] == "VERIFICATION_DECIDED"]),
                4,
            )

    def test_invalid_review_bundle_leaves_no_partial_decisions_or_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            decisions = [
                {"candidate_id": candidate.candidate_id, "status": "CONFIRMED"}
                for candidate in compiled.review_queue
            ]
            decisions[-1] = {
                "candidate_id": compiled.review_queue[-1].candidate_id,
                "status": "CORRECTED",
                "corrected_value": {"side": "LEFT", "mortise": "MAINTAINED"},
            }
            invalid = root / "partial-invalid.json"
            invalid.write_text(
                json.dumps(
                    {
                        "reviewer": "LOCAL_USER",
                        "decided_at": "2026-08-18T12:00:00+09:00",
                        "decisions": decisions,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                projector.review_and_materialize(compiled, invalid)
            self.assertEqual(
                [
                    record
                    for record in ledger.records()
                    if record["kind"] in {"VERIFICATION_DECIDED", "CLINICAL_EVENT_PROMOTED"}
                ],
                [],
            )

    def test_initial_review_cannot_reference_a_nonexistent_predecessor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            payload = json.loads((FIXTURES / "index_review.json").read_text(encoding="utf-8"))
            payload["decisions"][0]["replaces_decision_id"] = "decision-does-not-exist"
            invalid = root / "bad-parent.json"
            invalid.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial verification"):
                projector.review_and_materialize(compiled, invalid)
            self.assertFalse(
                any(record["kind"] == "VERIFICATION_DECIDED" for record in ledger.records())
            )

    def test_feature_version_advances_independently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            ankle_module = importlib.import_module("medic.ankle")
            with patch.object(
                ankle_module, "FEATURE_COMPONENT_VERSION", "ankle-feature-builder/0.2.0"
            ):
                second = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            self.assertEqual(second.state.state_id, first.state.state_id)
            self.assertEqual(second.feature.version, 2)
            self.assertNotEqual(second.feature.feature_id, first.feature.feature_id)
            self.assertEqual(
                len([record for record in ledger.records() if record["kind"] == "VERIFICATION_DECIDED"]),
                4,
            )

    def test_claim_readiness_requires_typed_minimum_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            payload = json.loads((FIXTURES / "index_source.json").read_text(encoding="utf-8"))
            for fact in payload["facts"][:3]:
                fact["semantic_type"] = "SYMPTOM"
            invalid_source = root / "wrong-semantics.json"
            invalid_source.write_text(json.dumps(payload), encoding="utf-8")
            compiled = compiler.compile_json("synthetic-ankle-001", invalid_source)
            review = root / "wrong-semantics-review.json"
            review.write_text(
                json.dumps(
                    {
                        "reviewer": "LOCAL_USER",
                        "decided_at": "2026-08-18T12:00:00+09:00",
                        "decisions": [
                            {"candidate_id": candidate.candidate_id, "status": "CONFIRMED"}
                            for candidate in compiled.review_queue
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = projector.review_and_materialize(compiled, review)
            self.assertEqual(result.claim_view.readiness, "NOT_READY")
            self.assertEqual(
                set(result.claim_view.missing_requirements),
                {"index_injury", "index_orif", "postoperative_alignment"},
            )

    def test_concurrent_revisions_cannot_fork_the_decision_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            injury = first.user_decisions[0]
            paths = []
            for index, status in enumerate(("UNKNOWN", "CORRECTED")):
                entry = {
                    "candidate_id": injury.candidate_id,
                    "status": status,
                    "replaces_decision_id": injury.decision_id,
                }
                if status == "CORRECTED":
                    entry["corrected_value"] = {
                        "side": "RIGHT",
                        "pattern": "TRIMALLEOLAR_FRACTURE_DISLOCATION",
                    }
                path = root / f"revision-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "reviewer": "LOCAL_USER",
                            "decided_at": f"2026-08-19T12:0{index}:00+09:00",
                            "decisions": [entry],
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            outcomes = []

            def revise(path):
                try:
                    projector.review_and_materialize(compiled, path)
                    outcomes.append("SUCCESS")
                except ValueError:
                    outcomes.append("REJECTED")

            threads = [threading.Thread(target=revise, args=(path,)) for path in paths]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(outcomes), ["REJECTED", "SUCCESS"])
            decisions = [
                record["payload"]["decision"]
                for record in ledger.records()
                if record["kind"] == "VERIFICATION_DECIDED"
                and record["payload"]["decision"]["candidate_id"] == injury.candidate_id
            ]
            self.assertEqual(len(decisions), 2)

    def test_verification_change_detection_preserves_json_scalar_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _ledger, compiler, projector = self.build_pipeline(root)
            index = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            projector.review_and_materialize(index, FIXTURES / "index_review.json")
            followup = compiler.compile_json(
                "synthetic-ankle-001", FIXTURES / "followup_source.json"
            )
            projector.review_and_materialize(followup, FIXTURES / "followup_review.json")
            payload = json.loads((FIXTURES / "followup_review.json").read_text(encoding="utf-8"))
            payload["decisions"] = [payload["decisions"][1]]
            payload["decisions"][0]["corrected_value"]["severity_0_10"] = True
            changed = root / "typed-change.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "replaces_decision_id"):
                projector.review_and_materialize(followup, changed)

    def test_reason_fresh_reuses_snapshot_but_creates_new_analysis_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")
            service = reasoning_module.ReasoningService(ledger)
            reasoner = reasoning_module.SyntheticReasoner()

            first = service.analyze(
                "synthetic-ankle-001", materialized.claim_view.claim_code, reasoner
            )
            second = service.analyze(
                "synthetic-ankle-001", materialized.claim_view.claim_code, reasoner
            )

            self.assertEqual(first.snapshot_id, second.snapshot_id)
            self.assertNotEqual(first.analysis_run_id, second.analysis_run_id)
            self.assertEqual(first.labels, ("SYNTHETIC", "NON-CLINICAL"))
            self.assertEqual(first.output["medical_conclusion_generated"], False)
            runs = [record for record in ledger.records() if record["kind"] == "ANALYSIS_RUN_COMPLETED"]
            self.assertEqual(len(runs), 2)

    def test_cli_runs_ingest_review_and_two_fresh_analyses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cli = importlib.import_module("medic.cli")
            outputs = []
            commands = (
                [
                    "ingest",
                    "--case-id",
                    "synthetic-ankle-001",
                    "--source",
                    str(FIXTURES / "index_source.json"),
                    "--store",
                    str(root),
                ],
                [
                    "review",
                    "--case-id",
                    "synthetic-ankle-001",
                    "--decisions",
                    str(FIXTURES / "index_review.json"),
                    "--store",
                    str(root),
                ],
                [
                    "analyze",
                    "--case-id",
                    "synthetic-ankle-001",
                    "--claim",
                    "ANKLE_POST_ORIF_STRUCTURAL_STATE",
                    "--reasoner",
                    "synthetic",
                    "--store",
                    str(root),
                ],
                [
                    "analyze",
                    "--case-id",
                    "synthetic-ankle-001",
                    "--claim",
                    "ANKLE_POST_ORIF_STRUCTURAL_STATE",
                    "--reasoner",
                    "synthetic",
                    "--store",
                    str(root),
                ],
            )
            for command in commands:
                stream = StringIO()
                with redirect_stdout(stream):
                    self.assertEqual(cli.main(command), 0)
                outputs.append(json.loads(stream.getvalue()))

            self.assertEqual(outputs[0]["review_queue_count"], 3)
            self.assertEqual(outputs[1]["claim_readiness"], "READY")
            self.assertNotEqual(outputs[2]["analysis_run_id"], outputs[3]["analysis_run_id"])
            self.assertEqual(outputs[2]["snapshot_id"], outputs[3]["snapshot_id"])
            self.assertEqual(outputs[2]["labels"], ["SYNTHETIC", "NON-CLINICAL"])

    def test_snapshot_lineage_excludes_unrelated_and_unreviewed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")
            service = reasoning_module.ReasoningService(ledger)
            first = service.analyze(
                "synthetic-ankle-001", materialized.claim_view.claim_code,
                reasoning_module.SyntheticReasoner(),
            )
            unrelated_payload = json.loads(
                (FIXTURES / "followup_source.json").read_text(encoding="utf-8")
            )
            unrelated_payload["case_id"] = "synthetic-other-case"
            unrelated = root / "unrelated.json"
            unrelated.write_text(json.dumps(unrelated_payload), encoding="utf-8")
            compiler.compile_json("synthetic-other-case", unrelated)

            second = service.analyze(
                "synthetic-ankle-001", materialized.claim_view.claim_code,
                reasoning_module.SyntheticReasoner(),
            )

            self.assertEqual(second.snapshot_id, first.snapshot_id)

    def test_analysis_selects_latest_claim_projection_for_same_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            first = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            ankle_module = importlib.import_module("medic.ankle")
            with patch.object(
                ankle_module, "CLAIM_COMPONENT_VERSION", "ankle-claim-view/0.2.0"
            ):
                latest = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")
            run = reasoning_module.ReasoningService(ledger).analyze(
                "synthetic-ankle-001", latest.claim_view.claim_code,
                reasoning_module.SyntheticReasoner(),
            )

            self.assertNotEqual(latest.claim_view.claim_view_id, first.claim_view.claim_view_id)
            self.assertEqual(run.claim_view_id, latest.claim_view.claim_view_id)

    def test_reasoning_service_rejects_clinical_or_extra_provider_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")

            class UnsafeReasoner:
                name = "unsafe"
                version = "1"

                def reason(self, _claim_view):
                    return {
                        "medical_conclusion_generated": True,
                        "diagnosis": "forbidden",
                        "labels": ["CLINICAL"],
                    }

            with self.assertRaisesRegex(ValueError, "non-clinical contract"):
                reasoning_module.ReasoningService(ledger).analyze(
                    "synthetic-ankle-001", materialized.claim_view.claim_code,
                    UnsafeReasoner(),
                )
            self.assertFalse(
                any(record["kind"] == "ANALYSIS_RUN_COMPLETED" for record in ledger.records())
            )

    def test_reasoning_service_rejects_schema_valid_hidden_clinical_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")

            class HiddenClinicalReasoner:
                name = "hidden-clinical"
                version = "1"

                def reason(self, claim_view):
                    return {
                        "medical_conclusion_generated": False,
                        "claim_code": "DIAGNOSIS_CONFIRMED",
                        "readiness": claim_view.readiness.value,
                        "eligible_event_count": len(claim_view.eligible_event_ids),
                        "supporting_event_count": len(claim_view.supporting_event_ids),
                        "contradicting_event_count": len(claim_view.contradicting_event_ids),
                        "missing_requirements": ["Start treatment immediately"],
                        "notice": "Diagnosis confirmed; stop follow-up and medication.",
                    }

            with self.assertRaisesRegex(ValueError, "exact synthetic non-clinical"):
                reasoning_module.ReasoningService(ledger).analyze(
                    "synthetic-ankle-001", materialized.claim_view.claim_code,
                    HiddenClinicalReasoner(),
                )

    def test_reasoning_output_contract_rejects_coercible_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")

            class CoercibleReasoner:
                name = "coercible"
                version = "1"

                def reason(self, claim_view):
                    output = reasoning_module.expected_non_clinical_output(claim_view)
                    output["eligible_event_count"] = str(output["eligible_event_count"])
                    output["missing_requirements"] = tuple(output["missing_requirements"])
                    return output

            with self.assertRaisesRegex(ValueError, "non-clinical contract"):
                reasoning_module.ReasoningService(ledger).analyze(
                    "synthetic-ankle-001", materialized.claim_view.claim_code,
                    CoercibleReasoner(),
                )

    def test_snapshot_sealer_version_change_creates_new_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")
            service = reasoning_module.ReasoningService(ledger)
            first = service.analyze(
                "synthetic-ankle-001", materialized.claim_view.claim_code,
                reasoning_module.SyntheticReasoner(),
            )
            with patch.object(reasoning_module, "COMPONENT_VERSION", "reasoning-service/0.2.0"):
                second = service.analyze(
                    "synthetic-ankle-001", materialized.claim_view.claim_code,
                    reasoning_module.SyntheticReasoner(),
                )
            self.assertNotEqual(first.snapshot_id, second.snapshot_id)

    def test_review_and_analysis_share_the_same_case_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger, compiler, projector = self.build_pipeline(root)
            compiled = compiler.compile_json("synthetic-ankle-001", FIXTURES / "index_source.json")
            materialized = projector.review_and_materialize(compiled, FIXTURES / "index_review.json")
            reasoning_module = importlib.import_module("medic.reasoning")
            completed = []

            def analyze():
                reasoning_module.ReasoningService(ledger).analyze(
                    "synthetic-ankle-001", materialized.claim_view.claim_code,
                    reasoning_module.SyntheticReasoner(),
                )
                completed.append(True)

            with ledger.key_lock("case", "synthetic-ankle-001"):
                thread = threading.Thread(target=analyze)
                thread.start()
                time.sleep(0.05)
                self.assertEqual(completed, [])
            thread.join()
            self.assertEqual(completed, [True])


if __name__ == "__main__":
    unittest.main()
