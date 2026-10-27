from __future__ import annotations

import importlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


FIXTURE = Path(__file__).parent / "fixtures" / "clinical_compiler" / "index_source.json"


class CompilerPipelineTests(unittest.TestCase):
    def modules(self):
        try:
            return (
                importlib.import_module("medic.ledger"),
                importlib.import_module("medic.compiler"),
            )
        except ModuleNotFoundError as exc:
            self.fail(f"compiler pipeline modules must exist: {exc}")

    def build_pipeline(self, root: Path, expected_alias: str = "LOCAL-SELF-001"):
        ledger_module, compiler_module = self.modules()
        store = ledger_module.ContentAddressedStore(root)
        ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")
        compiler = compiler_module.ClinicalCompiler(
            store=store,
            ledger=ledger,
            expected_subject_alias=expected_alias,
        )
        return store, ledger, compiler

    def test_raw_registration_is_content_addressed_and_byte_preserving(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, _ledger, _compiler = self.build_pipeline(root)

            first = store.register_raw("synthetic-ankle-001", FIXTURE)
            second = store.register_raw("synthetic-ankle-001", FIXTURE)

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertEqual(first.content_hash, second.content_hash)
            stored_paths = list((root / "raw").iterdir())
            self.assertEqual(len(stored_paths), 1)
            self.assertEqual(stored_paths[0].read_bytes(), FIXTURE.read_bytes())

    def test_same_source_reuses_compile_stage_without_duplicate_ledger_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _store, ledger, compiler = self.build_pipeline(root)

            first = compiler.compile_json("synthetic-ankle-001", FIXTURE)
            second = compiler.compile_json("synthetic-ankle-001", FIXTURE)

            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertEqual(first.compile_id, second.compile_id)
            self.assertEqual(
                tuple(candidate.candidate_id for candidate in first.candidates),
                tuple(candidate.candidate_id for candidate in second.candidates),
            )
            completed = [record for record in ledger.records() if record["kind"] == "COMPILE_COMPLETED"]
            self.assertEqual(len(completed), 1)

    def test_compiled_candidates_are_source_linked_and_not_promoted_before_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _store, ledger, compiler = self.build_pipeline(root)

            result = compiler.compile_json("synthetic-ankle-001", FIXTURE)

            self.assertEqual(len(result.candidates), 4)
            self.assertEqual(len({candidate.candidate_id for candidate in result.candidates}), 4)
            for index, candidate in enumerate(result.candidates):
                self.assertEqual(candidate.source_locators[0].locator, f"/facts/{index}")
                self.assertEqual(candidate.source_locators[0].artifact_id, result.artifact.artifact_id)
            self.assertEqual(len(result.review_queue), 3)
            self.assertEqual(
                [record for record in ledger.records() if record["kind"] == "CLINICAL_EVENT_PROMOTED"],
                [],
            )

    def test_subject_mismatch_blocks_candidates_at_the_integrity_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mismatched = root / "mismatched.json"
            payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
            payload["subject"]["local_alias"] = "SOMEONE-ELSE"
            mismatched.write_text(json.dumps(payload), encoding="utf-8")
            _store, ledger, compiler = self.build_pipeline(root)

            result = compiler.compile_json("synthetic-ankle-001", mismatched)

            self.assertEqual(result.integrity.status, "MISMATCH")
            self.assertEqual(result.candidates, ())
            self.assertEqual(result.review_queue, ())
            completed = [record for record in ledger.records() if record["kind"] == "COMPILE_COMPLETED"]
            self.assertEqual(completed[0]["payload"]["integrity_status"], "MISMATCH")

    def test_compile_parses_the_registered_immutable_bytes_not_a_mutated_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mutable_source = root / "upload.json"
            mutable_source.write_bytes(FIXTURE.read_bytes())
            store, _ledger, compiler = self.build_pipeline(root)
            original_register = store.register_raw

            def register_then_mutate(case_id, source_path):
                artifact = original_register(case_id, source_path)
                payload = json.loads(mutable_source.read_text(encoding="utf-8"))
                payload["facts"][0]["value"]["side"] = "LEFT"
                mutable_source.write_text(json.dumps(payload), encoding="utf-8")
                return artifact

            with patch.object(store, "register_raw", side_effect=register_then_mutate):
                result = compiler.compile_json("synthetic-ankle-001", mutable_source)

            self.assertEqual(result.candidates[0].normalized_value["side"], "RIGHT")
            self.assertEqual(store.read_raw(result.artifact).read_bytes(), FIXTURE.read_bytes())

    def test_normalized_blocks_are_persisted_and_candidate_dependencies_resolve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _store, _ledger, compiler = self.build_pipeline(root)

            result = compiler.compile_json("synthetic-ankle-001", FIXTURE)

            self.assertEqual(len(result.blocks), 4)
            blocks = {block.block_id: block for block in result.blocks}
            for candidate in result.candidates:
                block_id = candidate.dependency_ids[0]
                self.assertIn(block_id, blocks)
                self.assertIn("effective_at", blocks[block_id].raw_payload)
            reused = compiler.compile_json("synthetic-ankle-001", FIXTURE)
            self.assertEqual(result.blocks, reused.blocks)

    def test_cache_hit_repairs_a_missing_completion_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, ledger, compiler = self.build_pipeline(root)
            first = compiler.compile_json("synthetic-ankle-001", FIXTURE)
            ledger.path.unlink()

            reused = compiler.compile_json("synthetic-ankle-001", FIXTURE)

            self.assertTrue(reused.reused)
            completed = [record for record in ledger.records() if record["kind"] == "COMPILE_COMPLETED"]
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]["payload"]["compile_id"], first.compile_id)
            self.assertTrue(store.compiled_path(first.compile_id).exists())

    def test_tampered_compile_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, _ledger, compiler = self.build_pipeline(root)
            result = compiler.compile_json("synthetic-ankle-001", FIXTURE)
            cache_path = store.compiled_path(result.compile_id)
            wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
            wrapper["manifest"]["candidates"][0]["normalized_value"]["side"] = "LEFT"
            wrapper["manifest"]["review_queue_ids"] = []
            ledger_module, _compiler_module = self.modules()
            wrapper["manifest_hash"] = ledger_module.sha256_digest(
                ledger_module.canonical_json_bytes(wrapper["manifest"])
            )
            cache_path.write_text(json.dumps(wrapper), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "compile cache integrity"):
                compiler.compile_json("synthetic-ankle-001", FIXTURE)

    def test_raw_metadata_cannot_substitute_a_different_blob(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, _ledger, _compiler = self.build_pipeline(root)
            first = store.register_raw("synthetic-ankle-001", FIXTURE)
            other_source = root / "other.json"
            other_source.write_text('{"different":true}', encoding="utf-8")
            other = store.register_raw("synthetic-ankle-001", other_source)
            metadata_path = root / "metadata" / f"{first.artifact_id}.json"
            substituted = first.model_dump(mode="json")
            substituted["artifact_id"] = other.artifact_id
            substituted["content_hash"] = other.content_hash
            metadata_path.write_text(json.dumps(substituted), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "artifact metadata integrity"):
                store.register_raw("synthetic-ankle-001", FIXTURE)

    def test_completion_is_not_recorded_when_cache_commit_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, ledger, compiler = self.build_pipeline(root)
            with patch.object(store, "save_compiled", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    compiler.compile_json("synthetic-ankle-001", FIXTURE)

            self.assertEqual(ledger.records(), [])

    def test_missing_ledger_recovery_rejects_noncanonical_review_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, ledger, compiler = self.build_pipeline(root)
            result = compiler.compile_json("synthetic-ankle-001", FIXTURE)
            ledger.path.unlink()
            cache_path = store.compiled_path(result.compile_id)
            wrapper = json.loads(cache_path.read_text(encoding="utf-8"))
            wrapper["manifest"]["review_queue_ids"].append("candidate-does-not-exist")
            ledger_module, _compiler_module = self.modules()
            wrapper["manifest_hash"] = ledger_module.sha256_digest(
                ledger_module.canonical_json_bytes(wrapper["manifest"])
            )
            cache_path.write_text(json.dumps(wrapper), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "review queue"):
                compiler.compile_json("synthetic-ankle-001", FIXTURE)

    def test_source_schema_rejects_string_boolean_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
            payload["facts"][0]["high_impact"] = "false"
            payload["facts"][0]["unexpected"] = "not allowed"
            invalid = root / "invalid.json"
            invalid.write_text(json.dumps(payload), encoding="utf-8")
            _store, _ledger, compiler = self.build_pipeline(root)

            with self.assertRaisesRegex(ValueError, "source JSON schema"):
                compiler.compile_json("synthetic-ankle-001", invalid)

    def test_ledger_append_once_is_thread_safe_and_detects_existing_conflicts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_module, _compiler_module = self.modules()
            ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")
            threads = [
                threading.Thread(
                    target=ledger.append_once,
                    args=("TEST", "same-key", {"value": 1}),
                )
                for _ in range(12)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(ledger.records()), 1)

            record = ledger.records()[0]
            conflicting = dict(record)
            conflicting["payload"] = {"value": 2}
            with ledger.path.open("a", encoding="utf-8") as ledger_file:
                ledger_file.write(json.dumps(conflicting) + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate or conflicting ledger"):
                ledger.append_once("TEST", "same-key", {"value": 1})

    def test_same_source_compiles_safely_in_parallel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _store, ledger, compiler = self.build_pipeline(root)
            results = []
            errors = []

            def compile_source():
                try:
                    results.append(compiler.compile_json("synthetic-ankle-001", FIXTURE))
                except Exception as exc:  # assertions inspect every worker failure
                    errors.append(exc)

            threads = [threading.Thread(target=compile_source) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 12)
            self.assertEqual(len({result.artifact.artifact_id for result in results}), 1)
            self.assertEqual(len({result.compile_id for result in results}), 1)
            completed = [record for record in ledger.records() if record["kind"] == "COMPILE_COMPLETED"]
            self.assertEqual(len(completed), 1)

    def test_ledger_recomputes_record_ids_and_enforces_natural_key_uniqueness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_module, _compiler_module = self.modules()
            ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")
            ledger.path.write_text(
                json.dumps(
                    {
                        "record_id": "record-forged",
                        "kind": "TEST",
                        "natural_key": "same-key",
                        "payload": {"value": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "ledger corruption"):
                ledger.records()

    def test_ledger_rejects_invalid_records_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_module, _compiler_module = self.modules()
            ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")

            for kind, natural_key, payload in (("", "key", {}), ("TEST", "", {}), ("TEST", "key", [])):
                with self.subTest(kind=kind, natural_key=natural_key, payload=payload):
                    with self.assertRaisesRegex(ValueError, "ledger record"):
                        ledger.append_once(kind, natural_key, payload)
            self.assertFalse(ledger.path.exists())

    def test_ledger_rejects_json_inputs_that_change_type_on_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_module, _compiler_module = self.modules()
            ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")

            for payload in ({1: "integer-key"}, {"tuple": (1, 2)}):
                with self.subTest(payload=payload):
                    with self.assertRaisesRegex(ValueError, "strict JSON"):
                        ledger.append_once("TEST", "key", payload)
            self.assertFalse(ledger.path.exists())

    def test_ledger_conflict_comparison_preserves_json_scalar_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_module, _compiler_module = self.modules()
            ledger = ledger_module.AppendOnlyJsonlLedger(root / "ledger.jsonl")
            ledger.append_once("TEST", "typed-key", {"value": 1})

            for different in (True, 1.0):
                with self.subTest(different=different):
                    with self.assertRaisesRegex(RuntimeError, "append-only ledger conflict"):
                        ledger.append_once("TEST", "typed-key", {"value": different})


if __name__ == "__main__":
    unittest.main()
