"""Synthetic, offline acceptance tests for the bounded M2 context slice."""
import argparse
import importlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from medic.ledger import AppendOnlyJsonlLedger, canonical_json_bytes, deterministic_id, sha256_digest


class CaseContextTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("medic.case_context")
        except ModuleNotFoundError:
            self.fail("the bounded case context API must exist")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = AppendOnlyJsonlLedger(self.root / "ledger.jsonl")
        self.service = self.api.CaseContextService(self.ledger)
        self.scope = {"case_id": "synthetic-case", "patient_id": "synthetic-patient",
                      "episode_id": "episode-a", "epoch_id": "baseline",
                      "anatomy": "ankle", "laterality": "right"}
        self.facts = [self.fact("fact-a"), self.fact("fact-b"), self.fact("fact-c")]

    def fact(self, fact_id, **updates):
        return {"fact_id": fact_id, "context": dict(self.scope), **updates}

    def profile(self, **updates):
        return {"schema_version": "case-profile/1", "case_id": "synthetic-case",
                "patient_id": "synthetic-patient",
                "symptoms": [{"name": "pain", "value": "after walking",
                              "missingness": "present", "fact_ids": ["fact-a"]}],
                "function": [], "activity": [], "constraints": [], "goals": [],
                "priorities": [], **updates}

    def review(self, fact_id="fact-a", **updates):
        return {"schema_version": "source-input-review/1", "fact_id": fact_id,
                "context": dict(self.scope), "classification": "observation",
                "missingness": "present", "dates": [], **updates}

    def save_profile(self, **kwargs):
        return self.service.save_profile("synthetic-case", self.profile(),
            available_facts=self.facts, reviewer="synthetic-reviewer", reason="baseline", **kwargs)

    def save_review(self, data=None, **kwargs):
        return self.service.review_input("synthetic-case", data or self.review(),
            available_facts=self.facts, reviewer="synthetic-reviewer", reason="source checked", **kwargs)

    def test_profile_roundtrip_is_content_hashed_immutable_and_idempotent(self):
        first = self.save_profile()
        self.assertEqual(first.version, 1)
        self.assertEqual(first.event_id, deterministic_id("case-context", first.model_dump(
            mode="json", exclude={"event_id"})))
        before = self.ledger.path.read_bytes()
        self.assertEqual(self.save_profile(), first)
        self.assertEqual(self.ledger.path.read_bytes(), before)
        loaded = self.api.CaseContextService(self.ledger).status("synthetic-case")
        self.assertEqual(loaded.profile, first)
        self.assertEqual(loaded.review_inputs, ())
        self.assertEqual(loaded.event_count, 1)
        with self.assertRaises(ValidationError):
            first.profile.patient_id = "changed"
        with self.assertRaises(TypeError):
            first.profile.symptoms[0].fact_ids[0] = "changed"

    def test_profile_has_all_six_dimensions_and_preserves_order(self):
        item = {"name": "synthetic", "value": "documented", "missingness": "present"}
        data = self.profile(**{key: [item] for key in
            ("symptoms", "function", "activity", "constraints", "goals", "priorities")})
        data["priorities"] = [dict(item, name="mobility"), dict(item, name="comfort")]
        event = self.service.save_profile("synthetic-case", data,
            reviewer="reviewer", reason="reviewed")
        self.assertEqual([p.name for p in event.profile.priorities], ["mobility", "comfort"])
        for key in ("symptoms", "function", "activity", "constraints", "goals", "priorities"):
            broken = dict(data)
            del broken[key]
            with self.subTest(key=key), self.assertRaises(ValidationError):
                self.api.CaseProfile.model_validate(broken)

    def test_amend_and_withdraw_preserve_history_without_resurrecting_old_profile(self):
        first = self.save_profile()
        prefix = self.ledger.path.read_bytes()
        second = self.service.save_profile("synthetic-case", self.profile(goals=[{
            "name": "walking", "value": "synthetic goal", "missingness": "present"}]),
            available_facts=self.facts, reviewer="reviewer", reason="goal clarified",
            replaces=first.event_id)
        self.assertEqual(second.version, 2)
        self.assertEqual(second.replaces, first.event_id)
        withdrawn = self.service.withdraw("synthetic-case", second.event_id,
            reviewer="reviewer", reason="entered in error")
        self.assertEqual(withdrawn.version, 3)
        self.assertIsNone(self.service.status("synthetic-case").profile)
        self.assertTrue(self.ledger.path.read_bytes().startswith(prefix))
        self.assertEqual(self.service.history("synthetic-case"), (first, second, withdrawn))
        self.assertEqual(self.service.withdraw("synthetic-case", second.event_id,
            reviewer="reviewer", reason="entered in error"), withdrawn)
        replacement = self.save_profile(replaces=withdrawn.event_id)
        self.assertEqual(replacement.version, 4)

    def test_stale_wrong_case_unknown_and_implicit_replacements_fail_without_append(self):
        first = self.save_profile()
        second = self.save_profile(replaces=first.event_id)
        before = self.ledger.path.read_bytes()
        for replaces in (None, first.event_id, "unknown"):
            with self.subTest(replaces=replaces), self.assertRaises(ValueError):
                self.service.save_profile("synthetic-case", self.profile(goals=[{
                    "name": "new", "value": "new", "missingness": "present"}]),
                    available_facts=self.facts, reviewer="reviewer", reason="new",
                    replaces=replaces)
        for case_id, target in (("another-case", second.event_id),
                                ("synthetic-case", first.event_id),
                                ("synthetic-case", "unknown")):
            with self.subTest(case_id=case_id, target=target), self.assertRaises(ValueError):
                self.service.withdraw(case_id, target, reviewer="reviewer", reason="withdraw")
        self.assertEqual(self.ledger.path.read_bytes(), before)

    def test_competing_amendments_allow_only_one_successor(self):
        first = self.save_profile()
        def amend(index):
            try:
                return self.service.save_profile("synthetic-case", self.profile(goals=[{
                    "name": "goal", "value": str(index), "missingness": "present"}]),
                    available_facts=self.facts, reviewer="reviewer", reason="amended",
                    replaces=first.event_id)
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(amend, range(2)))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(len(self.service.history("synthetic-case")), 2)

    def test_each_source_classification_is_explicit_and_frozen(self):
        for classification in ("observation", "clinician_assertion", "request", "procedure",
                               "self_report", "actual_exposure", "derived_measure"):
            extra = {}
            if classification == "actual_exposure":
                extra["exposure_basis"] = "documented_administration"
            if classification == "derived_measure":
                extra["derived_from_ids"] = ["fact-b"]
            with self.subTest(classification=classification):
                review = self.api.SourceInputReview.model_validate(
                    self.review(classification=classification, **extra))
                self.assertEqual(review.classification, classification)
                with self.assertRaises(ValidationError):
                    review.classification = "request"

    def test_request_and_prescription_never_become_actual_exposure_implicitly(self):
        event = self.save_review(self.review(classification="request"))
        self.assertFalse(event.review.actual_exposure)
        for data in (self.review(classification="actual_exposure"),
                     self.review(classification="request", exposure_basis="documented_administration"),
                     self.review(classification="prescription")):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                self.api.SourceInputReview.model_validate(data)
        prescription = [self.fact("fact-a", source_kind="prescription")]
        with self.assertRaises(ValueError):
            self.service.review_input("synthetic-case", self.review(classification="actual_exposure",
                exposure_basis="documented_administration"), available_facts=prescription,
                reviewer="reviewer", reason="bad promotion", replaces=event.event_id)

    def test_derived_and_clinical_duplicate_links_do_not_change_byte_identity(self):
        same_hash = "sha256:" + "a" * 64
        self.facts = [self.fact("fact-a", content_hash=same_hash),
                      self.fact("fact-b", content_hash=same_hash)]
        first = self.save_review()
        other = self.save_review(self.review("fact-b"))
        self.assertEqual(len(self.service.status("synthetic-case").review_inputs), 2)
        derived = self.save_review(self.review(classification="derived_measure",
            derived_from_ids=["fact-b"]), replaces=first.event_id)
        self.assertFalse(derived.review.independent_evidence)
        for relation, independent in (("copied", False), ("corrected", False), ("true_repeat", True)):
            data = self.review(duplicate={"relation": relation, "fact_id": "fact-b"})
            event = self.save_review(data, replaces=derived.event_id)
            self.assertEqual(event.review.independent_evidence, independent)
            derived = event
        self.assertEqual(self.service.status("synthetic-case").review_inputs[1], other)

    def test_derived_review_requires_lineage_and_prohibits_self_links(self):
        for data in (self.review(classification="derived_measure"),
                     self.review(derived_from_ids=["fact-a"]),
                     self.review(derived_from_ids=["fact-b", "fact-b"]),
                     self.review(duplicate={"relation": "copied", "fact_id": "fact-a"})):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                self.api.SourceInputReview.model_validate(data)

    def test_reference_validation_rejects_unknown_other_case_and_patient(self):
        for data in (self.review("missing"), self.review(derived_from_ids=["missing"]),
                     self.review(duplicate={"relation": "copied", "fact_id": "missing"})):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.save_review(data)
        for field, value in (("case_id", "other-case"), ("patient_id", "other-patient")):
            foreign = self.fact("fact-b", context={**self.scope, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.service.review_input("synthetic-case", self.review(derived_from_ids=["fact-b"]),
                    available_facts=[self.facts[0], foreign], reviewer="reviewer", reason="invalid")
        with self.assertRaises(ValueError):
            self.service.save_profile("synthetic-case", self.profile(),
                reviewer="reviewer", reason="unknown profile reference")
        with self.assertRaises(ValueError):
            self.service.review_input("synthetic-case", self.review(),
                available_facts=[self.facts[0], self.facts[0]], reviewer="reviewer", reason="ambiguous IDs")
        self.assertEqual(self.ledger.records(), [])

    def test_context_dimensions_cannot_overwrite_known_primary_fact_identity(self):
        for field, value in (("case_id", "other"), ("patient_id", "other"),
                             ("episode_id", "other"), ("epoch_id", "followup"),
                             ("anatomy", "knee"), ("laterality", "left")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.save_review(self.review(context={**self.scope, field: value}))
        event = self.save_review()
        self.assertEqual(event.review.context.episode_id, "episode-a")
        self.assertEqual(event.review.context.epoch_id, "baseline")

    def test_partial_date_roles_and_precision_are_retained_without_imputing_days(self):
        dates = [{"role": "documentation", "precision": "month", "value": "2026-02"},
                 {"role": "event", "precision": "year", "value": "2025"},
                 {"role": "acquisition", "precision": "day", "value": "2026-02-28"},
                 {"role": "onset", "precision": "unknown", "value": None}]
        event = self.save_review(self.review(dates=dates))
        self.assertEqual([d.model_dump() for d in event.review.dates], dates)
        for value, precision in (("2026-02-30", "day"), ("2026-13", "month"),
                                 ("2026-02", "day"), ("2026", "unknown"),
                                 (None, "year"), ("0000", "year")):
            with self.subTest(value=value, precision=precision), self.assertRaises(ValidationError):
                self.api.SourceInputReview.model_validate(self.review(dates=[{
                    "role": "event", "precision": precision, "value": value}]))

    def test_authoring_and_issuance_dates_remain_distinct_through_ledger_replay(self):
        dates = [{"role": "authoring", "precision": "day", "value": "2026-01-10"},
                 {"role": "issuance", "precision": "day", "value": "2026-01-12"},
                 {"role": "documentation", "precision": "month", "value": "2026-01"},
                 {"role": "acquisition", "precision": "day", "value": "2026-01-09"},
                 {"role": "event", "precision": "year", "value": "2025"}]
        try:
            event = self.save_review(self.review(dates=dates))
        except ValidationError:
            self.fail("authoring and issuance must be accepted as separate partial date roles")
        loaded = self.api.CaseContextService(self.ledger).status("synthetic-case").review_inputs[0]
        self.assertEqual(event, loaded)
        self.assertEqual([item.model_dump() for item in loaded.review.dates], dates)

    def test_explicit_negative_is_not_missing_and_missingness_is_required(self):
        states = ("explicit_negative", "not_provided", "not_measured", "not_assessable")
        for state in states:
            item = {"name": "swelling", "missingness": state}
            event = self.api.CaseProfile.model_validate(self.profile(symptoms=[item]))
            self.assertEqual(event.symptoms[0].missingness, state)
            self.assertIsNone(event.symptoms[0].value)
        for item in ({"name": "pain", "missingness": "present"},
                     {"name": "pain", "value": "none", "missingness": "not_provided"},
                     {"name": "pain", "value": "unknown"}):
            with self.subTest(item=item), self.assertRaises(ValidationError):
                self.api.CaseProfile.model_validate(self.profile(symptoms=[item]))

    def test_review_amendment_withdrawal_does_not_hide_profile_or_other_inputs(self):
        profile = self.save_profile()
        first = self.save_review()
        other = self.save_review(self.review("fact-b"))
        changed = self.save_review(self.review(classification="clinician_assertion"), replaces=first.event_id)
        self.service.withdraw("synthetic-case", changed.event_id, reviewer="reviewer", reason="withdrawn")
        status = self.service.status("synthetic-case")
        self.assertEqual(status.profile, profile)
        self.assertEqual(status.review_inputs, (other,))
        self.assertEqual(len(self.service.history("synthetic-case")), 5)

    def test_unknown_schema_extra_fields_and_blank_audit_fields_fail_closed(self):
        for data in (self.profile(schema_version="case-profile/2"),
                     self.profile(api_key="synthetic-secret")):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                self.api.CaseProfile.model_validate(data)
        for reviewer, reason in (("", "reason"), ("reviewer", " ")):
            with self.subTest(reviewer=reviewer), self.assertRaises(ValueError):
                self.service.save_profile("synthetic-case", self.profile(),
                    available_facts=self.facts, reviewer=reviewer, reason=reason)
        self.assertEqual(self.ledger.records(), [])

    def test_enrichment_is_separate_from_confirmed_fact_contracts_and_excludes_inactive_versions(self):
        self.facts = [self.fact(fact_id, value_sha256=sha256_digest(canonical_json_bytes("synthetic")))
                      for fact_id in ("fact-a", "fact-b")]
        first = self.save_profile()
        reviewed = self.save_review(self.review(classification="derived_measure", derived_from_ids=["fact-b"]))
        base = {"confirmed_facts": ["synthetic source text"], "fact_details": [{
            "fact_id": "fact-a", "fact_key": "pain", "value": "synthetic",
            "user_confirmed": True, "scope": {"laterality": "right"}}, {
            "fact_id": "fact-b", "fact_key": "synthetic", "value": "synthetic",
            "user_confirmed": True}], "uncertainties": []}
        before = json.loads(json.dumps(base))
        enriched = self.api.enrich_context(self.ledger, "synthetic-case", base)
        self.assertEqual(base, before)
        self.assertEqual(enriched["fact_details"], before["fact_details"])
        extension = enriched["case_context"]
        self.assertEqual(extension["profile"]["event_id"], first.event_id)
        self.assertEqual(extension["review_inputs"][0]["review"]["derived_from_ids"], ["fact-b"])
        self.assertEqual(extension["review_inputs"][0]["event_id"], reviewed.event_id)
        enriched["fact_details"][0]["value"] = "mutated copy"
        self.assertEqual(base, before)
        self.service.withdraw("synthetic-case", reviewed.event_id, reviewer="reviewer", reason="withdrawn")
        self.assertEqual(self.api.enrich_context(self.ledger, "synthetic-case", base)
                         ["case_context"]["review_inputs"], [])

    def test_unpinned_direct_service_review_is_auditable_but_not_usable_at_runtime(self):
        event = self.save_review()
        context = {"fact_details": [{"fact_id": "fact-a", "value": "synthetic", "user_confirmed": True}]}
        before = self.ledger.path.read_bytes()
        extension = self.api.enrich_context(self.ledger, "synthetic-case", context)["case_context"]
        self.assertEqual(extension["review_inputs"], [])
        self.assertEqual(extension["stale_input_review_ids"], [event.event_id])
        self.assertEqual(extension["stale_input_review_reasons"][event.event_id], [
            {"fact_id": "fact-a", "reason": "VALUE_HASH_UNPINNED"}])
        self.assertEqual(self.service.status("synthetic-case").review_inputs, (event,))
        self.assertEqual(self.ledger.path.read_bytes(), before)

    def test_legacy_events_without_value_hash_remain_readable(self):
        event = self.save_review()
        self.assertNotIn("value_sha256", json.loads(self.ledger.path.read_text(encoding="utf-8"))
                         ["payload"]["available_facts"][0])
        self.assertEqual(self.service.history("synthetic-case"), (event,))

    def test_runtime_validates_dependency_and_duplicate_values_without_cli_recursion(self):
        self.facts = [self.fact(fact_id, value_sha256=sha256_digest(canonical_json_bytes("old")))
                      for fact_id in ("fact-a", "fact-b")]
        for relation in ("derived", "copied"):
            data = self.review(**({"derived_from_ids": ["fact-b"]} if relation == "derived" else
                                 {"duplicate": {"relation": "copied", "fact_id": "fact-b"}}))
            head = self.service.status("synthetic-case").review_inputs
            event = self.save_review(data, **({"replaces": head[0].event_id} if head else {}))
            context = {"fact_details": [
                {"fact_id": "fact-a", "value": "old", "user_confirmed": True},
                {"fact_id": "fact-b", "value": "revised", "user_confirmed": True}]}
            with patch("medic.cli._latest_context", side_effect=AssertionError("recursive context lookup")):
                extension = self.api.enrich_context(self.ledger, "synthetic-case", context)["case_context"]
            with self.subTest(relation=relation):
                self.assertEqual(extension["review_inputs"], [])
                self.assertEqual(extension["stale_input_review_reasons"][event.event_id], [
                    {"fact_id": "fact-b", "reason": "VALUE_CHANGED"}])

    def test_runtime_marks_missing_unreviewed_ambiguous_or_valueless_facts_stale(self):
        self.facts = [self.fact("fact-a", value_sha256=sha256_digest(canonical_json_bytes("old")))]
        event = self.save_review()
        actual = {"fact_id": "fact-a", "value": "old", "user_confirmed": True}
        cases = (([], "FACT_NOT_AVAILABLE"), ([{**actual, "user_confirmed": False}], "FACT_NOT_AVAILABLE"),
                 ([actual, actual], "AMBIGUOUS_FACT_ID"),
                 ([{"fact_id": "fact-a", "user_confirmed": True}], "CURRENT_VALUE_UNAVAILABLE"))
        for details, reason in cases:
            with self.subTest(reason=reason):
                extension = self.api.enrich_context(self.ledger, "synthetic-case", {
                    "fact_details": details})["case_context"]
                self.assertEqual(extension["review_inputs"], [])
                self.assertEqual(extension["stale_input_review_reasons"][event.event_id], [
                    {"fact_id": "fact-a", "reason": reason}])

    def test_profile_linked_fact_staleness_is_flagged_without_rewriting_profile(self):
        self.facts = [self.fact("fact-a", value_sha256=sha256_digest(canonical_json_bytes("old")))]
        profile = self.save_profile()
        context = {"fact_details": [{"fact_id": "fact-a", "value": "new", "user_confirmed": True}]}
        extension = self.api.enrich_context(self.ledger, "synthetic-case", context)["case_context"]
        self.assertEqual(extension["profile"], profile.model_dump(mode="json"))
        self.assertTrue(extension["profile_stale"])
        self.assertEqual(extension["profile_stale_reasons"], [{"fact_id": "fact-a", "reason": "VALUE_CHANGED"}])
        self.assertEqual(self.service.history("synthetic-case"), (profile,))

    def test_preconstructed_models_are_revalidated_before_any_append(self):
        profile = self.api.CaseProfile.model_validate(self.profile()).model_copy(
            update={"schema_version": "case-profile/99"})
        with self.assertRaises(ValueError):
            self.service.save_profile("synthetic-case", profile, available_facts=self.facts,
                reviewer="reviewer", reason="bad schema")
        review = self.api.SourceInputReview.model_validate(self.review()).model_copy(
            update={"classification": "not-a-classification"})
        with self.assertRaises(ValueError):
            self.service.review_input("synthetic-case", review, available_facts=self.facts,
                reviewer="reviewer", reason="bad classification")
        self.assertEqual(self.ledger.records(), [])

    def test_source_review_unknown_patient_cannot_hide_cross_patient_lineage(self):
        data = self.review(derived_from_ids=["fact-b"])
        data["context"]["patient_id"] = None
        foreign = self.fact("fact-b", context={**self.scope, "patient_id": "foreign"})
        with self.assertRaises(ValueError):
            self.service.review_input("synthetic-case", data, available_facts=[self.facts[0], foreign],
                reviewer="reviewer", reason="cross patient")

    def test_admission_rejects_review_patient_conflicting_with_active_profile(self):
        self.save_profile()
        before = self.ledger.path.read_bytes()
        identity = {**self.scope, "patient_id": "other-patient"}
        with self.assertRaises(ValueError):
            self.service.review_input("synthetic-case", self.review(context=identity),
                available_facts=[self.fact("fact-a", context=identity)], reviewer="reviewer", reason="conflict")
        self.assertEqual(self.ledger.path.read_bytes(), before)

    def test_admission_rejects_new_profile_conflicting_with_active_reviewed_facts(self):
        self.save_review()
        before = self.ledger.path.read_bytes()
        with self.assertRaises(ValueError):
            self.service.save_profile("synthetic-case", self.profile(patient_id="other-patient", symptoms=[]),
                reviewer="reviewer", reason="conflict")
        self.assertEqual(self.ledger.path.read_bytes(), before)

    def test_profile_patient_change_requires_withdrawal_and_reconciling_active_facts(self):
        profile = self.save_profile()
        reviewed = self.save_review()
        changed = self.profile(patient_id="other-patient", symptoms=[])
        with self.assertRaises(ValueError):
            self.service.save_profile("synthetic-case", changed, reviewer="reviewer", reason="patient change",
                                      replaces=profile.event_id)
        withdrawn = self.service.withdraw("synthetic-case", profile.event_id, reviewer="reviewer", reason="reconcile")
        with self.assertRaises(ValueError):
            self.service.save_profile("synthetic-case", changed, reviewer="reviewer", reason="patient change",
                                      replaces=withdrawn.event_id)
        self.service.withdraw("synthetic-case", reviewed.event_id, reviewer="reviewer", reason="reconcile")
        new = self.service.save_profile("synthetic-case", changed, reviewer="reviewer", reason="patient change",
                                        replaces=withdrawn.event_id)
        self.assertEqual(new.profile.patient_id, "other-patient")

    def test_unknown_patient_policy_preserves_unknown_without_inference_or_false_conflict(self):
        self.save_profile()
        identity = {**self.scope, "patient_id": None}
        pin = sha256_digest(canonical_json_bytes("synthetic"))
        event = self.service.review_input("synthetic-case", self.review(context=identity),
            available_facts=[self.fact("fact-a", context=identity, value_sha256=pin)],
            reviewer="reviewer", reason="identity unknown")
        runtime = self.api.enrich_context(self.ledger, "synthetic-case", {"fact_details": [
            {"fact_id": "fact-a", "value": "synthetic", "user_confirmed": True}]})["case_context"]
        self.assertIsNone(event.review.context.patient_id)
        self.assertIsNone(runtime["review_inputs"][0]["review"]["context"]["patient_id"])
        self.assertEqual(runtime["ineligible_fact_ids"], [])
        self.assertEqual(runtime["patient_identity_policy"], "KNOWN_CONFLICTS_ONLY_UNKNOWN_UNRESOLVED")

    def test_runtime_quarantines_legacy_conflicting_review_but_preserves_raw_facts_and_history(self):
        self.save_profile()
        identity = {**self.scope, "patient_id": "other-patient"}
        legacy = self.api.CaseContextService(AppendOnlyJsonlLedger(self.root / "legacy.jsonl"))
        event = legacy.review_input("synthetic-case", self.review(context=identity),
            available_facts=[self.fact("fact-a", context=identity,
                value_sha256=sha256_digest(canonical_json_bytes("synthetic")))],
            reviewer="reviewer", reason="pre-guard legacy record")
        self.ledger.append_once("CASE_CONTEXT_VERSIONED", event.event_id, event.model_dump(mode="json"))
        before = self.ledger.path.read_bytes()
        base = {"fact_details": [{"fact_id": "fact-a", "value": "synthetic", "user_confirmed": True}]}
        result = self.api.enrich_context(self.ledger, "synthetic-case", base)
        runtime = result["case_context"]
        self.assertEqual(result["fact_details"], base["fact_details"])
        self.assertEqual(runtime["review_inputs"], [])
        self.assertEqual(runtime["ineligible_fact_ids"], ["fact-a"])
        self.assertIn("PATIENT_ID_CONFLICT", runtime["ineligible_fact_reasons"]["fact-a"])
        self.assertIn({"fact_id": "fact-a", "reason": "PATIENT_ID_CONFLICT"},
                      runtime["stale_input_review_reasons"][event.event_id])
        self.assertEqual(self.ledger.path.read_bytes(), before)
        self.assertEqual(len(self.service.history("synthetic-case")), 2)

    def test_runtime_source_patient_conflicts_exclude_ids_even_without_semantic_review(self):
        self.save_profile()
        for identity in ({"patient_id": "other-patient"}, {"person_id": "other-patient"}):
            base = {"fact_details": [{"fact_id": "source-b", "value": "synthetic",
                "user_confirmed": True, "scope": identity}], "imaging_reviews": [{
                    "review_status": "USER_REVIEWED", "measurements": [{"fact_id": "image-b", "value": 1,
                                                                         "metadata": identity}]}]}
            runtime = self.api.enrich_context(self.ledger, "synthetic-case", base)["case_context"]
            with self.subTest(identity=identity):
                self.assertEqual(runtime["ineligible_fact_ids"], ["image-b", "source-b"])

    def test_runtime_source_identity_change_invalidates_review_even_when_wording_matches(self):
        self.facts = [self.fact("fact-a", value_sha256=sha256_digest(canonical_json_bytes("synthetic")))]
        self.save_profile()
        event = self.save_review()
        base = {"fact_details": [{"fact_id": "fact-a", "value": "synthetic", "user_confirmed": True,
                                  "scope": {"patient_id": "other-patient"}}]}
        runtime = self.api.enrich_context(self.ledger, "synthetic-case", base)["case_context"]
        self.assertEqual(runtime["review_inputs"], [])
        self.assertEqual(runtime["stale_input_review_ids"], [event.event_id])
        self.assertEqual(runtime["ineligible_fact_ids"], ["fact-a"])
        self.assertTrue(runtime["profile_stale"])

    def test_historical_retry_after_withdrawal_never_reactivates_old_content(self):
        first = self.save_profile()
        self.service.withdraw("synthetic-case", first.event_id, reviewer="reviewer", reason="withdrawn")
        self.assertEqual(self.save_profile(), first)
        self.assertIsNone(self.service.status("synthetic-case").profile)
        self.assertEqual(self.service.status("synthetic-case").event_count, 2)

    def test_forged_chain_and_snapshot_are_rejected_even_with_valid_content_hash(self):
        first = self.save_profile()
        original = self.ledger.path.read_bytes()
        for change in ({"version": 2}, {"replaces": "missing"},
                       {"available_facts": []}, {"schema_version": "case-context/99"}):
            payload = {**first.model_dump(mode="json", exclude={"event_id"}), **change}
            event_id = deterministic_id("case-context", payload)
            record = self.ledger._prepare_record("CASE_CONTEXT_VERSIONED", event_id,
                                                 {**payload, "event_id": event_id})
            self.ledger.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.service.status("synthetic-case")
        self.ledger.path.write_bytes(original)

    def test_corrupt_hash_chain_schema_and_truncated_records_fail_closed(self):
        first = self.save_profile()
        original = self.ledger.path.read_text(encoding="utf-8")
        record = json.loads(original)
        record["payload"]["reason"] = "tampered"
        self.ledger.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.service.status("synthetic-case")
        self.ledger.path.write_text(original, encoding="utf-8")
        self.ledger.append_once("UNRELATED", "unrelated", {"case_id": "other-case"})
        self.assertEqual(self.service.status("synthetic-case").profile, first)
        with self.ledger.path.open("ab") as stream:
            stream.write(b'{"truncated":')
        with self.assertRaises(RuntimeError):
            self.service.status("synthetic-case")


class CaseCliTests(unittest.TestCase):
    def setUp(self):
        try:
            self.cli = importlib.import_module("medic.case_cli")
        except ModuleNotFoundError:
            self.fail("the case CLI registration hook must exist")
        self.parser = argparse.ArgumentParser()
        self.cli.register(self.parser.add_subparsers(dest="command", required=True))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    def args(self, action, *extra):
        return self.parser.parse_args(["case", action, "--case", "synthetic-case",
            "--store", str(self.root / "store"), *extra])

    def profile(self):
        return {"schema_version": "case-profile/1", "case_id": "synthetic-case",
                **{key: [] for key in ("symptoms", "function", "activity", "constraints", "goals", "priorities")}}

    def ledger(self):
        return AppendOnlyJsonlLedger(self.root / "store" / "ledger.jsonl")

    def seed_facts(self, facts, case_id="synthetic-case"):
        details = [{"fact_id": fact_id, "fact_key": "synthetic", "value": "synthetic",
                    "user_confirmed": True, "confirmation_scope": "SOURCE_TRANSCRIPTION_ONLY",
                    "scope": {}} if isinstance(fact_id, str) else fact_id for fact_id in facts]
        self.ledger().append_once("HIGH_IMPACT_FACTS_CONFIRMED", "confirmed-" + case_id,
            {"case_id": case_id, "context": {"fact_details": details}})

    def review_file(self, fact_id="fact-a", **updates):
        return self.file("review.json", {"schema_version": "source-input-review/1",
            "fact_id": fact_id, "context": {"case_id": "synthetic-case"},
            "classification": "observation", "missingness": "present", **updates})

    def review_args(self, filename, *extra):
        return self.args("review-input", "--file", filename, "--reviewer", "reviewer",
                         "--reason", "checked", *extra)

    def test_registration_is_default_local_and_dispatch_declines_unrelated_commands(self):
        args = self.parser.parse_args(["case", "status", "--case", "synthetic-case"])
        self.assertEqual(args.store, Path(".medic"))
        self.assertIsNone(self.cli.execute(argparse.Namespace(command="unrelated")))
        code, payload = self.cli.execute(self.args("status"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["event_count"], 0)
        self.assertIsNone(payload["profile"])

    def test_case_questions_registration_inherits_local_store_and_accepts_case_flag(self):
        with redirect_stderr(io.StringIO()):
            try:
                args = self.parser.parse_args(["case", "questions", "--case", "synthetic-case"])
            except SystemExit:
                self.fail("case CLI must register the questions adapter")
        self.assertEqual(args.case_id, "synthetic-case")
        self.assertEqual(args.store, Path(".medic"))
        args = self.parser.parse_args(["case", "--store", str(self.root / "custom"),
                                      "questions", "--case", "synthetic-case"])
        self.assertEqual(args.store, self.root / "custom")

    def test_case_questions_dispatches_before_m2_context_service_is_created(self):
        args = argparse.Namespace(command="case", case_action="questions", case_id="synthetic-case",
                                  store=self.root / "store")
        with patch("medic.case_cli.CaseContextService", side_effect=AssertionError("M2 service must not run")):
            try:
                code, payload = self.cli.execute(args)
            except AssertionError:
                self.fail("question commands must dispatch before constructing the M2 service")
        self.assertEqual(code, 0)
        self.assertEqual(payload["version"], 0)

    def test_parent_main_routes_question_file_and_status_through_case_adapter(self):
        self.seed_facts(["fact-a"])
        filename = self.file("questions.json", {"case_id": "synthetic-case", "reviewer": "reviewer",
            "reason": "checked", "questions": [{"question_id": "q-state", "type": "CURRENT_STATE",
                "text": "Synthetic state?", "target": "synthetic target", "assessment_time": "2026-01",
                "allowed_fact_ids": ["fact-a"]}]})
        cli_main = importlib.import_module("medic.cli")
        arguments = ["case", "questions", "--case", "synthetic-case", "--store", str(self.root / "store")]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            try:
                code = cli_main.main([*arguments, "--file", filename])
            except SystemExit:
                self.fail("parent main must parse case questions through the case adapter")
        self.assertEqual(code, 0)
        saved = json.loads(output.getvalue())
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["evaluations"][0]["status"], "ANSWER")
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli_main.main(arguments)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["event_id"], saved["event_id"])
        self.assertNotIn("Synthetic state?", output.getvalue())

    def test_profile_file_amend_withdraw_and_status_are_offline_and_json_serializable(self):
        filename = self.file("profile.json", self.profile())
        code, first = self.cli.execute(self.args("profile", "--file", filename,
            "--reviewer", "reviewer", "--reason", "baseline"))
        self.assertEqual(code, 0)
        code, amended = self.cli.execute(self.args("profile", "--file", filename,
            "--reviewer", "reviewer", "--reason", "rechecked", "--replaces", first["event_id"]))
        self.assertEqual(code, 0)
        self.assertEqual(amended["version"], 2)
        code, withdrawn = self.cli.execute(self.args("profile", "--withdraw", amended["event_id"],
            "--reviewer", "reviewer", "--reason", "withdrawn"))
        self.assertEqual(code, 0)
        code, status = self.cli.execute(self.args("status"))
        self.assertIsNone(status["profile"])
        self.assertEqual(status["event_count"], 3)
        json.dumps(withdrawn)

    def test_review_input_uses_actual_fact_ids_with_optional_user_context(self):
        context = {"case_id": "synthetic-case", "facts": [{"fact_id": "fact-a",
            "context": {"case_id": "synthetic-case"}}]}
        filename = self.file("review.json", {"schema_version": "source-input-review/1",
            "fact_id": "fact-a", "context": {"case_id": "synthetic-case"},
            "classification": "request", "missingness": "not_provided"})
        base = ("--file", filename, "--reviewer", "reviewer", "--reason", "checked")
        self.assertEqual(self.cli.execute(self.args("review-input", *base))[0], 2)
        self.seed_facts(["fact-a"])
        self.assertEqual(self.cli.execute(self.args("review-input", *base))[0], 0)
        context_file = self.file("context.json", context)
        code, event = self.cli.execute(self.args("review-input", *base, "--context", context_file))
        self.assertEqual(code, 0)
        self.assertFalse(event["actual_exposure"])
        code, _ = self.cli.execute(self.args("review-input", "--withdraw", event["event_id"],
            "--reviewer", "reviewer", "--reason", "withdrawn"))
        self.assertEqual(code, 0)

    def test_forged_context_cannot_whitelist_source_or_profile_references(self):
        context_file = self.file("forged-context.json", {"case_id": "synthetic-case", "facts": [
            {"fact_id": "forged-fact", "context": {"case_id": "synthetic-case"}}]})
        filename = self.review_file("forged-fact")
        code, payload = self.cli.execute(self.review_args(filename, "--context", context_file))
        self.assertEqual(code, 2)
        self.assertNotIn("forged-fact", json.dumps(payload))
        profile = self.profile()
        profile["symptoms"] = [{"name": "pain", "value": "synthetic", "missingness": "present",
                                "fact_ids": ["forged-fact"]}]
        filename = self.file("profile.json", profile)
        code, _ = self.cli.execute(self.args("profile", "--file", filename,
            "--context", context_file, "--reviewer", "reviewer", "--reason", "checked"))
        self.assertEqual(code, 2)
        self.assertEqual(self.ledger().records(), [])

    def test_user_context_must_be_a_subset_of_known_case_ids_even_when_unused(self):
        self.seed_facts(["fact-a"])
        self.seed_facts(["foreign-fact"], case_id="another-case")
        before = self.ledger().path.read_bytes()
        for extra in ("forged-fact", "foreign-fact"):
            context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
                {"fact_id": "fact-a", "context": {"case_id": "synthetic-case"}},
                {"fact_id": extra, "context": {"case_id": "synthetic-case"}}]})
            with self.subTest(extra=extra):
                code, _ = self.cli.execute(self.review_args(self.review_file(), "--context", context_file))
                self.assertEqual(code, 2)
                self.assertEqual(self.ledger().path.read_bytes(), before)

    def test_valid_scope_overlay_may_omit_other_real_lineage_ids(self):
        self.seed_facts(["fact-a", "fact-b"])
        context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
            {"fact_id": "fact-a", "context": {"case_id": "synthetic-case", "epoch_id": "baseline"}}]})
        filename = self.review_file(classification="derived_measure", derived_from_ids=["fact-b"],
            context={"case_id": "synthetic-case", "epoch_id": "baseline"})
        code, payload = self.cli.execute(self.review_args(filename, "--context", context_file))
        self.assertEqual(code, 0)
        self.assertFalse(payload["independent_evidence"])

    def test_enriched_context_snapshots_cannot_whitelist_their_own_fact_ids(self):
        self.seed_facts(["fact-a"])
        api = importlib.import_module("medic.case_context")
        identity = {"case_id": "synthetic-case"}
        api.CaseContextService(self.ledger()).review_input("synthetic-case", {
            "schema_version": "source-input-review/1", "fact_id": "forged-in-extension",
            "context": identity, "classification": "observation", "missingness": "present"},
            available_facts=[{"fact_id": "forged-in-extension", "context": identity}],
            reviewer="reviewer", reason="explicit direct API snapshot")
        cli_main = importlib.import_module("medic.cli")
        latest = cli_main._latest_context
        def enriched(ledger, case_id):
            return api.enrich_context(ledger, case_id, latest(ledger, case_id))
        context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
            {"fact_id": "forged-in-extension", "context": identity}]})
        before = self.ledger().path.read_bytes()
        with patch("medic.cli._latest_context", side_effect=enriched) as load:
            self.assertEqual(self.cli.execute(self.review_args(self.review_file("forged-in-extension"),
                "--context", context_file))[0], 2)
            load.assert_called_once()
        self.assertEqual(self.ledger().path.read_bytes(), before)

    def test_user_context_can_fill_unknown_identity_without_removing_known_scope(self):
        self.seed_facts([{"fact_id": "fact-a", "fact_key": "synthetic", "value": "synthetic",
            "user_confirmed": True, "scope": {"anatomical_site": "ankle", "laterality": "RIGHT"}}])
        identity = {"case_id": "synthetic-case", "patient_id": "synthetic-person",
                    "episode_id": "episode-a", "epoch_id": "baseline"}
        context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
            {"fact_id": "fact-a", "context": identity}]})
        filename = self.review_file(context={**identity, "anatomy": "ankle", "laterality": "right"})
        code, _ = self.cli.execute(self.review_args(filename, "--context", context_file))
        self.assertEqual(code, 0)
        event = importlib.import_module("medic.case_context").CaseContextService(
            self.ledger()).status("synthetic-case").review_inputs[0]
        self.assertEqual(event.available_facts[0].context.anatomy, "ankle")
        self.assertEqual(event.available_facts[0].context.laterality, "right")
        self.assertEqual(event.available_facts[0].context.patient_id, "synthetic-person")

    def test_user_context_cannot_conflict_with_any_known_identity_dimension(self):
        identity = {"case_id": "synthetic-case", "patient_id": "person-a", "episode_id": "episode-a",
                    "epoch_id": "baseline", "anatomy": "ankle", "laterality": "right"}
        self.seed_facts([{"fact_id": "fact-a", "fact_key": "synthetic", "value": "synthetic",
                         "user_confirmed": True, "scope": identity}])
        before = self.ledger().path.read_bytes()
        for field in identity:
            bad = {**identity, field: "left" if field == "laterality" else "other"}
            context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
                {"fact_id": "fact-a", "context": bad}]})
            with self.subTest(field=field):
                code, _ = self.cli.execute(self.review_args(self.review_file(context=bad),
                                                           "--context", context_file))
                self.assertEqual(code, 2)
                self.assertEqual(self.ledger().path.read_bytes(), before)

    def test_unreviewed_withdrawn_and_legacy_idless_facts_are_not_available(self):
        self.seed_facts([{"fact_id": "unreviewed", "fact_key": "synthetic", "value": "synthetic",
                         "user_confirmed": False, "scope": {}},
                        {"fact_id": "withdrawn", "fact_key": "synthetic", "value": "synthetic",
                         "user_confirmed": True, "scope": {}},
                        {"fact_key": "legacy", "value": "synthetic", "user_confirmed": True}])
        self.ledger().append_once("SOURCE_DECISION", "withdrawn", {"case_id": "synthetic-case",
            "fact_id": "withdrawn", "withdraw": True, "value": None})
        before = self.ledger().path.read_bytes()
        for fact_id in ("unreviewed", "withdrawn", "legacy"):
            context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
                {"fact_id": fact_id, "context": {"case_id": "synthetic-case"}}]})
            with self.subTest(fact_id=fact_id):
                self.assertEqual(self.cli.execute(self.review_args(self.review_file(fact_id),
                    "--context", context_file))[0], 2)
        self.assertEqual(self.ledger().path.read_bytes(), before)

    def test_user_context_cannot_override_source_kind_or_invent_byte_identity(self):
        self.seed_facts([{"fact_id": "fact-a", "fact_key": "synthetic", "value": "synthetic",
            "user_confirmed": True, "scope": {"source_kind": "prescription"}}])
        for update in ({"source_kind": "actual_exposure"}, {"content_hash": "sha256:" + "a" * 64}):
            context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
                {"fact_id": "fact-a", "context": {"case_id": "synthetic-case"}, **update}]})
            with self.subTest(update=update):
                self.assertEqual(self.cli.execute(self.review_args(self.review_file(),
                    "--context", context_file))[0], 2)
        self.assertEqual(self.cli.execute(self.review_args(self.review_file(
            classification="actual_exposure", exposure_basis="documented_administration")))[0], 2)

    def test_cli_pins_canonical_value_hash_separately_from_raw_content_hash(self):
        raw_hash = "sha256:" + "b" * 64
        value = {"wording": "synthetic 소견", "score": 0}
        self.seed_facts([{"fact_id": "fact-a", "fact_key": "synthetic", "value": value,
                         "user_confirmed": True, "content_hash": raw_hash}])
        context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
            {"fact_id": "fact-a", "context": {"case_id": "synthetic-case", "epoch_id": "baseline"}}]})
        filename = self.review_file(context={"case_id": "synthetic-case", "epoch_id": "baseline"})
        self.assertEqual(self.cli.execute(self.review_args(filename, "--context", context_file))[0], 0)
        snapshot = self.ledger().records()[-1]["payload"]["available_facts"][0]
        self.assertEqual(snapshot.get("value_sha256"), sha256_digest(canonical_json_bytes(value)))
        self.assertEqual(snapshot["content_hash"], raw_hash)

    def test_latest_context_excludes_changed_and_withdrawn_semantic_reviews_without_changing_history(self):
        self.seed_facts(["fact-a"])
        code, saved = self.cli.execute(self.review_args(self.review_file()))
        self.assertEqual(code, 0)
        cli_main = importlib.import_module("medic.cli")
        current = cli_main._latest_context(self.ledger(), "synthetic-case")["case_context"]
        self.assertEqual(current["review_inputs"][0]["event_id"], saved["event_id"])
        for value, withdraw in (("revised wording", False), (None, True)):
            self.ledger().append_once("SOURCE_DECISION", "revision-" + str(withdraw), {
                "case_id": "synthetic-case", "fact_id": "fact-a", "value": value, "withdraw": withdraw})
            before = self.ledger().path.read_bytes()
            extension = cli_main._latest_context(self.ledger(), "synthetic-case")["case_context"]
            self.assertEqual(extension["review_inputs"], [])
            self.assertEqual(extension["stale_input_review_ids"], [saved["event_id"]])
            self.assertEqual(extension["stale_input_review_reasons"][saved["event_id"]], [
                {"fact_id": "fact-a", "reason": "FACT_NOT_AVAILABLE" if withdraw else "VALUE_CHANGED"}])
            self.assertEqual(self.ledger().path.read_bytes(), before)

    def test_explicit_rereview_after_wording_change_restores_only_new_review(self):
        self.seed_facts(["fact-a"])
        filename = self.review_file()
        _, original = self.cli.execute(self.review_args(filename))
        self.ledger().append_once("SOURCE_DECISION", "revised", {"case_id": "synthetic-case",
            "fact_id": "fact-a", "value": "new wording", "withdraw": False})
        code, revised = self.cli.execute(self.review_args(filename, "--replaces", original["event_id"]))
        self.assertEqual(code, 0)
        extension = importlib.import_module("medic.cli")._latest_context(
            self.ledger(), "synthetic-case")["case_context"]
        self.assertEqual([event["event_id"] for event in extension["review_inputs"]], [revised["event_id"]])
        self.assertEqual(extension["stale_input_review_ids"], [])
        self.assertEqual(len(importlib.import_module("medic.case_context").CaseContextService(
            self.ledger()).history("synthetic-case")), 2)

    def test_active_imaging_measurement_fact_uses_reviewed_metadata_and_conditions(self):
        measurement = {"kind": "line", "value": 12.0, "units": "mm", "measurement_id": "display-label",
            "metadata": {"person_id": "person-a", "episode_id": "episode-a", "epoch_id": "baseline",
                         "anatomy": "ankle"}}
        review = {"study_uid": "study", "series_uid": "series", "review_status": "USER_REVIEWED",
            "review_status_provenance": "synthetic", "measurements": [measurement], "segmentations": [],
            "conditions": {"laterality": ["R"]}, "interpretation": "manual", "unmarked_tissue": "NOT_EVALUATED"}
        self.ledger().append_once("IMAGING_REVIEW_IMPORTED", "imaging-review", {
            "case_id": "synthetic-case", "review_id": "imaging-review", "review": review})
        fact_id = deterministic_id("imaging-observation", {"review_id": "imaging-review",
            "kind": "measurement", "index": 0, "observation": measurement})
        identity = {"case_id": "synthetic-case", "patient_id": "person-a", "episode_id": "episode-a",
                    "epoch_id": "baseline", "anatomy": "ankle", "laterality": "right"}
        filename = self.review_file(fact_id, context=identity)
        self.assertEqual(self.cli.execute(self.review_args(filename))[0], 0)
        self.ledger().append_once("IMAGING_REVIEW_WITHDRAWN", "withdraw-image", {
            "case_id": "synthetic-case", "review_id": "imaging-review"})
        context_file = self.file("context.json", {"case_id": "synthetic-case", "facts": [
            {"fact_id": fact_id, "context": identity}]})
        before = self.ledger().path.read_bytes()
        self.assertEqual(self.cli.execute(self.review_args(filename, "--context", context_file))[0], 2)
        self.assertEqual(self.ledger().path.read_bytes(), before)

    def test_validation_errors_never_echo_input_secret_values_or_paths(self):
        secret = "synthetic-SECRET-value"
        filename = self.file("private-name.json", dict(self.profile(), api_key=secret))
        args = self.args("profile", "--file", filename, "--reviewer", "reviewer", "--reason", "checked")
        code, payload = self.cli.execute(args)
        self.assertEqual(code, 2)
        self.assertNotIn(secret, json.dumps(payload))
        self.assertNotIn(filename, json.dumps(payload))
        self.assertFalse((self.root / "store" / "ledger.jsonl").exists())
        Path(filename).write_text('{"schema_version": NaN}', encoding="utf-8")
        self.assertEqual(self.cli.execute(args)[0], 2)
        Path(filename).write_text('{"case_id": "one", "case_id": "two"}', encoding="utf-8")
        self.assertEqual(self.cli.execute(args)[0], 2)

    def test_mutation_flags_are_exclusive_and_wrong_category_withdrawal_rejected(self):
        filename = self.file("profile.json", self.profile())
        _, event = self.cli.execute(self.args("profile", "--file", filename,
            "--reviewer", "reviewer", "--reason", "checked"))
        code, _ = self.cli.execute(self.args("review-input", "--withdraw", event["event_id"],
            "--reviewer", "reviewer", "--reason", "wrong category"))
        self.assertEqual(code, 2)
        args = self.args("profile", "--withdraw", event["event_id"], "--replaces", "anything",
                         "--reviewer", "reviewer", "--reason", "invalid")
        self.assertEqual(self.cli.execute(args)[0], 2)

    def test_status_summaries_never_echo_free_text_or_audit_values(self):
        secret = "synthetic-SECRET-value"
        profile = self.profile()
        profile["goals"] = [{"name": "goal", "value": secret, "missingness": "present"}]
        filename = self.file("profile.json", profile)
        code, payload = self.cli.execute(self.args("profile", "--file", filename,
            "--reviewer", secret, "--reason", secret))
        self.assertEqual(code, 0)
        self.assertNotIn(secret, json.dumps(payload))
        _, status = self.cli.execute(self.args("status"))
        self.assertNotIn(secret, json.dumps(status))

    def test_case_level_store_flag_and_integrity_errors_are_supported(self):
        args = self.parser.parse_args(["case", "--store", str(self.root / "custom"),
                                      "status", "--case", "synthetic-case"])
        self.assertEqual(args.store, self.root / "custom")
        (self.root / "custom").mkdir(exist_ok=True)
        (self.root / "custom" / "ledger.jsonl").write_text("private-synthetic-corruption", encoding="utf-8")
        code, payload = self.cli.execute(args)
        self.assertEqual(code, 2)
        self.assertEqual(payload["error"], "case_context_integrity_error")
        self.assertNotIn("private-synthetic-corruption", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
