"""Synthetic offline acceptance tests for the bounded M3 questions slice."""
import argparse
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from medic.ledger import AppendOnlyJsonlLedger, deterministic_id


def test_questions_module_is_available():
    assert importlib.util.find_spec("medic.questions") is not None, "M3 bounded questions API is missing"


@pytest.fixture
def api():
    try:
        return importlib.import_module("medic.questions")
    except ModuleNotFoundError as exc:
        if exc.name != "medic.questions":
            raise
        pytest.fail("M3 requires the bounded medic.questions API")


@pytest.fixture
def available(api):
    return api.AvailableReferences(fact_ids=frozenset({"fact-1"}),
        measurement_ids=frozenset({"measure-1"}),
        evidence_ids=frozenset({"evidence-1", "validation", "calibration", "method"}))


def question(**updates):
    return {"question_id": "q-state", "type": "CURRENT_STATE", "text": "Synthetic state?",
            "target": "synthetic target", "assessment_time": "2026-01",
            "allowed_fact_ids": ("fact-1",), **updates}


def claim(**updates):
    return {"claim_id": "claim-1", "question_id": "q-state", "text": "Synthetic observation.",
            "status": "ANSWER", "citations": {"fact_ids": ("fact-1",)}, **updates}


def causal_question(**updates):
    return question(**{"question_id": "q-effect", "type": "TREATMENT_EFFECT", "horizon": "one year",
        "outcome": "synthetic outcome", "intervention": "synthetic A", "comparator": "synthetic B",
        "time_zero": "2026-01", **updates})


def numeric_claim(kind="ABSOLUTE_CAUSAL_EFFECT", **updates):
    return claim(question_id="q-effect", numeric_estimate={"kind": kind, "value": 0.1},
        citations={"fact_ids": ("fact-1",),
                   "evidence_ids": ("validation", "calibration", "method")},
        numeric_support={"validated_model_id": "synthetic-model/1", "model_validated": True,
            "validation_source_id": "validation", "calibration_source_id": "calibration",
            "method_source_id": "method", "scenario": {
                "target": "synthetic target", "assessment_time": "2026-01", "horizon": "one year",
                "outcome": "synthetic outcome", "intervention": "synthetic A",
                "comparator": "synthetic B", "time_zero": "2026-01"}}, **updates)


def bundle(**updates):
    return {"case_id": "synthetic-case", "reviewer": "synthetic-reviewer", "reason": "reviewed",
            "questions": (question(),), **updates}


def service(api, tmp_path):
    return api.CaseQuestionService(AppendOnlyJsonlLedger(tmp_path / "ledger.jsonl"))


def test_contract_is_versioned_strict_and_deeply_immutable(api):
    from medic.research_contracts import StrictFrozenModel
    original = question()
    q = api.QuestionContract.model_validate(original)
    assert isinstance(q, StrictFrozenModel)
    assert q.schema_version == "question/1"
    assert q.question_id == original["question_id"]
    with pytest.raises(ValidationError):
        q.question_id = "rewritten"
    with pytest.raises(TypeError):
        q.allowed_fact_ids[0] = "rewritten"
    for patch in ({"schema_version": "question/9"}, {"unknown": True}, {"question_id": 1},
                  {"question_id": " "}, {"allowed_fact_ids": (" fact-1 ",)},
                  {"allowed_fact_ids": ("fact-1", "fact-1")}, {"target": 123}):
        with pytest.raises(ValidationError):
            api.QuestionContract.model_validate({**original, **patch})
    with pytest.raises(ValidationError):
        api.QuestionContract(type="CURRENT_STATE", text="No generated question IDs")


@pytest.mark.parametrize("kind,missing", [
    ("CURRENT_STATE", ("target", "assessment_time", "observations_or_measurements")),
    ("PROGNOSIS", ("target", "assessment_time", "horizon", "outcome", "treatment_state")),
    ("TREATMENT_EFFECT", ("target", "assessment_time", "horizon", "outcome", "intervention",
                          "comparator", "time_zero")),
    ("DECISION", ("target", "assessment_time", "options", "goals", "constraints", "priorities")),
])
def test_missingness_is_deterministic_per_question_without_imputation(api, available, kind, missing):
    q = api.QuestionContract(question_id="explicit-id", type=kind, text="Synthetic question")
    result = api.evaluate_question(q, available=available)
    assert result.status == "MISSING_DATA"
    assert result.missing_fields == missing
    assert result == api.evaluate_question(q, available=available)
    assert q.assessment_time is None


def test_missing_preferences_do_not_block_current_state_or_conditional_options(api, available):
    decision = question(question_id="q-decision", type="DECISION", options=("A", "B"))
    results = api.evaluate_questions((question(), decision), available=available)
    assert [r.status for r in results] == ["ANSWER", "MISSING_DATA"]
    assert results[1].missing_fields == ("goals", "constraints", "priorities")
    options = claim(claim_id="options", question_id="q-decision", status="CONDITIONAL",
                    decision_mode="OPTIONS")
    single = claim(claim_id="single", question_id="q-decision")
    checked = api.validate_question_claims((question(), decision),
        (claim(), single, options), available=available)
    assert [c.accepted for c in checked] == [True, False, True]
    assert checked[1].status == "MISSING_DATA"
    complete = {**decision, "goals": ("mobility",), "constraints": (), "priorities": ("mobility",)}
    assert api.evaluate_question(complete, available=available).status == "ANSWER"


@pytest.mark.parametrize("field,value,reason", [
    ("allowed_fact_ids", ("unknown",), "unknown_fact_reference:unknown"),
    ("allowed_measurement_ids", ("unknown",), "unknown_measurement_reference:unknown"),
    ("allowed_evidence_ids", ("unknown",), "unknown_evidence_reference:unknown"),
])
def test_available_sets_are_independent_and_unknown_references_rejected(api, available, field, value, reason):
    result = api.evaluate_question(question(**{field: value}), available=available)
    assert result.status == "CONFLICT"
    assert reason in result.reasons


@pytest.mark.parametrize("axis", ["data", "measurement", "model", "literature", "applicability"])
def test_claim_has_each_explicit_uncertainty_axis(api, axis):
    c = api.QuestionClaim.model_validate(claim())
    assert getattr(c.uncertainty, axis) == "NOT_ASSESSED"
    with pytest.raises(ValidationError):
        setattr(c.uncertainty, axis, "changed")


@pytest.mark.parametrize("status", ["ANSWER", "CONDITIONAL", "MISSING_DATA", "OUT_OF_SCOPE",
                                   "CONFLICT", "SYSTEM_FAILURE"])
def test_all_claim_statuses_are_preserved(api, available, status):
    result, = api.validate_question_claims((question(),), (claim(status=status),), available=available)
    assert result.status == status
    assert result.accepted


def test_each_bad_claim_has_its_own_reasons_without_abstaining_other_claims(api, available):
    claims = (claim(), claim(claim_id="bad-ref", citations={"fact_ids": ("not-present",)}),
              claim(claim_id="bad-question", question_id="unknown"),
              claim(claim_id="bad-shape", status="MADE_UP"))
    results = api.validate_question_claims((question(),), claims, available=available)
    assert [r.accepted for r in results] == [True, False, False, False]
    assert "unknown_fact_reference:not-present" in results[1].reasons
    assert results[2].status == "OUT_OF_SCOPE"
    assert results[3].status == "SYSTEM_FAILURE"
    assert results[3].reasons == ("invalid_claim_contract",)


def test_known_but_disallowed_claim_reference_is_rejected(api, available):
    c = claim(citations={"evidence_ids": ("evidence-1",)})
    result, = api.validate_question_claims((question(),), (c,), available=available)
    assert not result.accepted
    assert "disallowed_evidence_reference:evidence-1" in result.reasons


def test_duplicate_question_and_claim_ids_cannot_be_ambiguous(api, available):
    with pytest.raises(ValueError, match="question"):
        api.evaluate_questions((question(), question()), available=available)
    results = api.validate_question_claims((question(),), (claim(), claim()), available=available)
    assert all(not r.accepted and "duplicate_claim_id" in r.reasons for r in results)


@pytest.mark.parametrize("kind", ["INDIVIDUAL_RISK", "NNT", "ABSOLUTE_CAUSAL_EFFECT"])
def test_numeric_claim_without_explicit_sources_is_withheld_locally(api, available, kind):
    bad = claim(claim_id="unsafe-number", numeric_estimate={"kind": kind, "value": 8.0 if kind == "NNT" else 0.2})
    results = api.validate_question_claims((question(),), (claim(), bad), available=available)
    assert results[0].accepted
    assert not results[1].accepted
    assert results[1].status == "MISSING_DATA"
    assert "numeric_support_required" in results[1].reasons


@pytest.mark.parametrize("text", ["Individual risk is 20%.", "NNT = 8", "Absolute risk reduction is 0.1",
                                 "The chance is 1 in 10.", "위험은 20%입니다."])
def test_numeric_text_cannot_bypass_the_structured_gate(api, available, text):
    result, = api.validate_question_claims((question(),), (claim(text=text),), available=available)
    assert not result.accepted
    assert "untyped_numeric_estimate" in result.reasons


def test_validated_numeric_scenario_is_explicit_matching_and_cited(api, available):
    q = causal_question(allowed_evidence_ids=("validation", "calibration", "method"))
    valid = numeric_claim()
    assert api.validate_question_claims((q,), (valid,), available=available)[0].accepted
    for field in ("validated_model_id", "validation_source_id", "calibration_source_id", "method_source_id"):
        support = {**valid["numeric_support"], field: None}
        result, = api.validate_question_claims((q,), ({**valid, "numeric_support": support},), available=available)
        assert not result.accepted
        assert "numeric_support_missing:" + field in result.reasons
    for support in ({**valid["numeric_support"], "model_validated": False},
                    {**valid["numeric_support"], "scenario": {**valid["numeric_support"]["scenario"], "comparator": "other"}},
                    {**valid["numeric_support"], "calibration_source_id": "not-known"}):
        result, = api.validate_question_claims((q,), ({**valid, "numeric_support": support},), available=available)
        assert not result.accepted
    result, = api.validate_question_claims((q,),
        ({**valid, "citations": {"fact_ids": ("fact-1",)}},), available=available)
    assert not result.accepted
    assert "numeric_source_not_cited:calibration" in result.reasons


def test_individual_risk_requires_treatment_state_and_horizon(api, available):
    q = question(question_id="q-effect", type="PROGNOSIS", horizon="one year", outcome="synthetic outcome",
                 treatment_state="synthetic unchanged", allowed_evidence_ids=("validation", "calibration", "method"))
    c = numeric_claim("INDIVIDUAL_RISK")
    support = c["numeric_support"]
    c["numeric_support"] = {**support, "scenario": {**support["scenario"], "treatment_state": "synthetic unchanged"}}
    assert api.validate_question_claims((q,), (c,), available=available)[0].accepted
    c["numeric_support"]["scenario"].pop("treatment_state")
    result, = api.validate_question_claims((q,), (c,), available=available)
    assert "numeric_scenario_missing:treatment_state" in result.reasons


def test_additional_test_requires_decision_change_burden_urgency_and_alternatives(api):
    data = {"proposal_id": "test-1", "question_id": "q-state", "test": "Synthetic test",
            "question": "What uncertainty will this resolve?",
            "results_change": ({"result": "positive", "change": "reconsider option A"},),
            "burdens": ("synthetic burden",), "urgency": "routine", "alternatives": ("defer",)}
    proposal = api.AdditionalTestProposal.model_validate(data)
    assert proposal.results_change[0].change == "reconsider option A"
    for key in ("question", "results_change", "burdens", "urgency", "alternatives"):
        invalid = {k: v for k, v in data.items() if k != key}
        with pytest.raises(ValidationError):
            api.AdditionalTestProposal.model_validate(invalid)


def test_save_load_revision_withdrawal_are_append_only_and_audited(api, available, tmp_path):
    svc = service(api, tmp_path)
    first = svc.save(bundle(), available=available)
    prefix = svc.ledger.path.read_bytes()
    assert first.version == 1
    assert first.event_id == deterministic_id("case-questions", first.model_dump(mode="json", exclude={"event_id"}))
    assert svc.save(bundle(), available=available) == first
    assert svc.ledger.path.read_bytes() == prefix
    second = svc.save(bundle(questions=(question(text="Revised synthetic question?"),),
                             replaces=first.event_id), available=available)
    assert second.version == 2
    assert second.bundle.questions[0].question_id == first.bundle.questions[0].question_id
    assert second.bundle.questions[0].allowed_fact_ids == ("fact-1",)
    last = svc.withdraw("synthetic-case", second.event_id, reviewer="reviewer", reason="entered in error")
    assert last.version == 3
    assert svc.load("synthetic-case").active is None
    assert svc.history("synthetic-case") == (first, second, last)
    assert svc.ledger.path.read_bytes().startswith(prefix)
    assert svc.withdraw("synthetic-case", second.event_id, reviewer="reviewer", reason="entered in error") == last
    assert svc.save(bundle(), available=available) == first
    assert svc.load("synthetic-case").active is None
    restored = svc.save(bundle(replaces=last.event_id), available=available)
    assert restored.version == 4


def test_stale_replacements_unknown_ids_and_audit_errors_append_nothing(api, available, tmp_path):
    svc = service(api, tmp_path)
    first = svc.save(bundle(), available=available)
    before = svc.ledger.path.read_bytes()
    for patch in ({"reason": " "}, {"reviewer": " "}, {"case_id": " "},
                  {"questions": (question(allowed_fact_ids=("invented",)),), "replaces": first.event_id},
                  {"questions": (question(text="changed"),)}, {"replaces": "stale"}):
        with pytest.raises((ValueError, ValidationError)):
            svc.save(bundle(**patch), available=available)
    for case_id, target in (("other-case", first.event_id), ("synthetic-case", "unknown")):
        with pytest.raises(ValueError):
            svc.withdraw(case_id, target, reviewer="reviewer", reason="withdraw")
    assert svc.ledger.path.read_bytes() == before


def test_invalid_numeric_claim_does_not_discard_saved_current_state(api, available, tmp_path):
    svc = service(api, tmp_path)
    bad = claim(claim_id="bad", numeric_estimate={"kind": "INDIVIDUAL_RISK", "value": 0.3})
    svc.save(bundle(claims=(claim(), bad)), available=available)
    projection = svc.load("synthetic-case")
    assert projection.evaluations[0].status == "ANSWER"
    assert [c.accepted for c in projection.claim_results] == [True, False]
    fresh = svc.load("synthetic-case", available=api.AvailableReferences())
    assert fresh.evaluations[0].status == "CONFLICT"
    assert all(not r.accepted for r in fresh.claim_results)


def test_malformed_claim_is_quarantined_without_blocking_bundle_questions(api, available, tmp_path):
    svc = service(api, tmp_path)
    invalid = claim(claim_id="bad-schema", status="INVALID", extra={"values": [1, 2]})
    data = json.loads(json.dumps(bundle(claims=(claim(), invalid))))
    event = svc.save(data, available=available)
    loaded = svc.load("synthetic-case")
    assert loaded.evaluations[0].status == "ANSWER"
    assert [result.accepted for result in loaded.claim_results] == [True, False]
    assert loaded.claim_results[1].reasons == ("invalid_claim_contract",)
    with pytest.raises(TypeError):
        event.bundle.claims[1]["extra"]["values"].append(3)


def test_reference_snapshot_pins_only_used_ids_and_retry_ignores_unrelated_sources(api, available, tmp_path):
    svc = service(api, tmp_path)
    first = svc.save(bundle(), available=available)
    assert first.available.fact_ids == frozenset({"fact-1"})
    assert first.available.measurement_ids == frozenset()
    assert first.available.evidence_ids == frozenset()
    more = api.AvailableReferences(fact_ids=frozenset({"fact-1", "unrelated"}))
    assert svc.save(bundle(), available=more) == first
    assert len(svc.history("synthetic-case")) == 1


@pytest.mark.parametrize("kind,value", [("INDIVIDUAL_RISK", 0.1), ("NNT", 8.0), ("ABSOLUTE_CAUSAL_EFFECT", -0.1)])
def test_complete_numeric_support_requires_compatible_question_type(api, available, kind, value):
    q = causal_question(type="CURRENT_STATE", treatment_state="synthetic unchanged",
                        allowed_evidence_ids=("validation", "calibration", "method"))
    c = numeric_claim(kind)
    c["numeric_estimate"]["value"] = value
    c["numeric_support"]["scenario"]["treatment_state"] = "synthetic unchanged"
    result, = api.validate_question_claims((q,), (c,), available=available)
    assert not result.accepted
    assert "numeric_question_type_mismatch" in result.reasons


def test_additional_proposal_references_are_validated_before_append(api, available, tmp_path):
    svc = service(api, tmp_path)
    proposal = {"proposal_id": "test-1", "question_id": "q-state", "test": "Synthetic test",
        "question": "Resolve uncertainty?", "results_change": ({"result": "A", "change": "revisit A"},),
        "burdens": ("time",), "urgency": "routine", "alternatives": ("defer",)}
    first = svc.save(bundle(additional_tests=(proposal,)), available=available)
    assert first.bundle.additional_tests[0].proposal_id == "test-1"
    before = svc.ledger.path.read_bytes()
    for change in ({"question_id": "unknown"}, {"citations": {"measurement_ids": ("measure-1",)}},
                   {"citations": {"evidence_ids": ("unknown",)}}):
        with pytest.raises(ValueError):
            svc.save(bundle(additional_tests=({**proposal, **change},), replaces=first.event_id), available=available)
    assert svc.ledger.path.read_bytes() == before


def test_competing_revisions_have_one_successor(api, available, tmp_path):
    svc = service(api, tmp_path)
    first = svc.save(bundle(), available=available)
    def amend(index):
        try:
            return svc.save(bundle(questions=(question(text=str(index)),), replaces=first.event_id), available=available)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(amend, (1, 2)))
    assert sum(result is not None for result in results) == 1
    assert len(svc.history("synthetic-case")) == 2


def test_mutated_prebuilt_contracts_are_revalidated_before_save(api, available, tmp_path):
    svc = service(api, tmp_path)
    q = api.QuestionContract.model_validate(question())
    object.__setattr__(q, "allowed_fact_ids", ("unknown",))
    with pytest.raises(ValueError):
        svc.save(bundle(questions=(q,)), available=available)
    assert svc.history("synthetic-case") == ()


@pytest.mark.parametrize("tamper", ["hash", "version", "references", "truncate"])
def test_corruption_is_detected_on_load(api, available, tmp_path, tamper):
    svc = service(api, tmp_path)
    svc.save(bundle(), available=available)
    record = json.loads(svc.ledger.path.read_text(encoding="utf-8"))
    if tamper == "hash":
        record["payload"]["bundle"]["questions"][0]["text"] = "tampered"
    elif tamper == "version":
        record["payload"]["version"] = 4
    elif tamper == "references":
        record["payload"]["available"]["fact_ids"] = []
    if tamper in {"version", "references"}:
        payload = record["payload"]
        payload["event_id"] = deterministic_id("case-questions", {k: v for k, v in payload.items() if k != "event_id"})
        record["natural_key"] = payload["event_id"]
        record["record_id"] = deterministic_id("record", {"kind": record["kind"], "natural_key": record["natural_key"]})
    svc.ledger.path.write_text(json.dumps(record) + ("\n{" if tamper == "truncate" else "\n"), encoding="utf-8")
    with pytest.raises(RuntimeError):
        svc.load("synthetic-case")


def parser(api):
    root = argparse.ArgumentParser()
    case = root.add_subparsers(dest="command").add_parser("case")
    case.add_argument("--store", default=".medic")
    api.register_questions(case.add_subparsers(dest="case_action"))
    return root


def test_cli_hook_declines_unrelated_commands_and_uses_local_default(api):
    assert api.execute_questions(SimpleNamespace(command="doctor")) is None
    assert api.execute_questions(SimpleNamespace(command="case", case_action="profile")) is None
    args = parser(api).parse_args(["case", "questions", "--case", "synthetic-case"])
    assert str(args.store) == ".medic"


def test_cli_import_is_lazy_and_file_status_are_offline(api, tmp_path, monkeypatch):
    calls = []
    def context(ledger, case_id):
        calls.append(case_id)
        return {"fact_details": [{"fact_id": "fact-1", "user_confirmed": True}],
                "imaging_reviews": [{"review_status": "USER_REVIEWED",
                                     "measurements": [{"fact_id": "measure-1"}]}]}
    monkeypatch.setitem(sys.modules, "medic.cli", SimpleNamespace(_latest_context=context))
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle()), encoding="utf-8")
    arguments = ["case", "questions", "--case", "synthetic-case", "--store", str(tmp_path)]
    code, payload = api.execute_questions(parser(api).parse_args(arguments + ["--file", str(path)]))
    assert code == 0
    assert payload["evaluations"][0]["status"] == "ANSWER"
    code, loaded = api.execute_questions(parser(api).parse_args(arguments))
    assert code == 0
    assert loaded["version"] == payload["version"] == 1
    assert len(calls) == 2
    serialized = json.dumps(loaded)
    for free_text in ("Synthetic state?", "synthetic-reviewer", "reviewed", str(path)):
        assert free_text not in serialized


def test_cli_cannot_whitelist_bundle_ids_or_echo_validation_secrets(api, tmp_path, monkeypatch):
    def missing(*args):
        raise KeyError("no reviewed context")
    monkeypatch.setitem(sys.modules, "medic.cli", SimpleNamespace(_latest_context=missing))
    path = tmp_path / "secret-input.json"
    path.write_text(json.dumps(bundle(reason="secret-reason")), encoding="utf-8")
    args = parser(api).parse_args(["case", "questions", "--case", "synthetic-case", "--file", str(path), "--store", str(tmp_path)])
    code, payload = api.execute_questions(args)
    assert code == 2
    assert not (tmp_path / "ledger.jsonl").exists()
    assert "secret" not in json.dumps(payload)
    assert "fact-1" not in json.dumps(payload)
    path.write_text('{"case_id":"synthetic-case","case_id":"secret"}', encoding="utf-8")
    assert api.execute_questions(args)[0] == 2


def test_cli_empty_status_and_case_mismatch_are_safe(api, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "medic.cli", SimpleNamespace(_latest_context=lambda *_: {}))
    arguments = ["case", "--store", str(tmp_path), "questions", "--case", "synthetic-case"]
    code, payload = api.execute_questions(parser(api).parse_args(arguments))
    assert code == 0 and payload["version"] == 0
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle(case_id="other-case")), encoding="utf-8")
    assert api.execute_questions(parser(api).parse_args(arguments + ["--file", str(path)]))[0] == 2


@pytest.mark.parametrize("source", ["reviewed", "unreviewed", "withdrawn", "legacy-idless", "other-case"])
def test_cli_validates_real_reviewed_fact_context_without_inventing_ids(api, tmp_path, source):
    ledger = AppendOnlyJsonlLedger(tmp_path / "ledger.jsonl")
    detail = {"fact_id": "fact-1", "fact_key": "synthetic", "value": "synthetic",
              "user_confirmed": source != "unreviewed", "scope": {}}
    if source == "legacy-idless":
        detail.pop("fact_id")
    ledger.append_once("HIGH_IMPACT_FACTS_CONFIRMED", "synthetic-confirmation",
        {"case_id": "other-case" if source == "other-case" else "synthetic-case",
         "context": {"fact_details": [detail]}})
    if source == "withdrawn":
        ledger.append_once("SOURCE_DECISION", "synthetic-withdrawal",
            {"case_id": "synthetic-case", "fact_id": "fact-1", "withdraw": True, "value": None})
    prefix = ledger.path.read_bytes()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle()), encoding="utf-8")
    args = parser(api).parse_args(["case", "questions", "--case", "synthetic-case", "--store", str(tmp_path), "--file", str(path)])
    code, payload = api.execute_questions(args)
    assert code == (0 if source == "reviewed" else 2)
    if source == "reviewed":
        assert payload["evaluations"][0]["status"] == "ANSWER"
        assert api.CaseQuestionService(ledger).load("synthetic-case").active.bundle.questions[0].allowed_fact_ids == ("fact-1",)
    else:
        assert ledger.path.read_bytes() == prefix


@pytest.mark.parametrize("mode", ["stable-id", "expert-reviewed", "display-label", "unreviewed", "withdrawn"])
def test_cli_measurement_ids_are_reviewed_observation_ids_not_display_labels(api, tmp_path, mode):
    ledger = AppendOnlyJsonlLedger(tmp_path / "ledger.jsonl")
    observation = {"measurement_id": "display-label", "value": 12.0, "units": "mm"}
    review = {"study_uid": "study", "series_uid": "series",
        "review_status": "UNREVIEWED" if mode == "unreviewed" else "EXPERT_REVIEWED" if mode == "expert-reviewed" else "USER_REVIEWED",
        "review_status_provenance": "synthetic", "measurements": [observation], "segmentations": [],
        "interpretation": "synthetic", "unmarked_tissue": "NOT_EVALUATED"}
    ledger.append_once("IMAGING_REVIEW_IMPORTED", "synthetic-review", {
        "case_id": "synthetic-case", "review_id": "synthetic-review", "review": review})
    stable_id = deterministic_id("imaging-observation", {"review_id": "synthetic-review",
        "kind": "measurement", "index": 0, "observation": observation})
    if mode == "withdrawn":
        ledger.append_once("IMAGING_REVIEW_WITHDRAWN", "withdraw-review", {
            "case_id": "synthetic-case", "review_id": "synthetic-review"})
    prefix = ledger.path.read_bytes()
    reference = "display-label" if mode == "display-label" else stable_id
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle(questions=(question(allowed_fact_ids=(),
        allowed_measurement_ids=(reference,)),))), encoding="utf-8")
    args = parser(api).parse_args(["case", "questions", "--case", "synthetic-case", "--store", str(tmp_path), "--file", str(path)])
    code, payload = api.execute_questions(args)
    assert code == (0 if mode in {"stable-id", "expert-reviewed"} else 2)
    if mode in {"stable-id", "expert-reviewed"}:
        assert payload["evaluations"][0]["status"] == "ANSWER"
        assert ledger.path.read_bytes().startswith(prefix)
        assert ledger.records()[0]["payload"]["review"] == review
    else:
        assert ledger.path.read_bytes() == prefix
