from __future__ import annotations

import argparse
import getpass
import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .ankle import ClinicalAnkleProjector, ReviewBundle
from .compiler import ClinicalCompiler
from .config import CostEstimate, ModelRegistry
from .evidence import (
    ClinicalTrialsSource,
    EuropePmcSource,
    EvidenceBundle,
    LiteratureSearchPlanner,
    OfficialLiteratureRetriever,
    OfficialAnkleGuidelineSource,
    PubMedSource,
)
from .extraction import ClinicalExtractionService, ExtractionResult, extraction_batches
from .intake import FileIntakeResult, FileIntakeService, ImageRole
from .ledger import AppendOnlyJsonlLedger, ContentAddressedStore, deterministic_id
from .providers import build_live_gateways
from .reasoning import ReasoningService, SyntheticReasoner
from .reporting import ReportRenderer
from .research_contracts import EvidenceCard, QuestionSet
from .tribunal import PROMPT_VERSION, TribunalResult, TribunalService
from .workflow import ResearchWorkflow
from .extended_cli import register, execute, budget_arguments, approved_estimate, read_json, run_id


def _components(root: Path):
    store = ContentAddressedStore(root)
    ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
    compiler = ClinicalCompiler(store, ledger, "LOCAL-SELF-001")
    return ledger, compiler, ClinicalAnkleProjector(ledger)


def _store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=Path("medic-data"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--case-id", required=True)
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--store", type=Path, required=True)

    ingest_files = subparsers.add_parser("ingest-files")
    ingest_files.add_argument("case")
    ingest_files.add_argument("paths", type=Path, nargs="+")
    ingest_files.add_argument("--reference-images", action="store_true")
    ingest_files.add_argument("--pdf-password-prompt", action="store_true")
    _store_argument(ingest_files)

    review = subparsers.add_parser("review")
    review.add_argument("case", nargs="?")
    review.add_argument("--case-id")
    review.add_argument("--decisions", type=Path)
    review.add_argument("--compile-id")
    review.add_argument("--interactive", action="store_true")
    review.add_argument("--approve-cost", action="store_true")
    review.add_argument("--input-tokens", type=int, default=10_000)
    review.add_argument("--store", type=Path, required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--case-id", required=True)
    analyze.add_argument("--claim", required=True)
    analyze.add_argument("--reasoner", choices=("synthetic",), required=True)
    analyze.add_argument("--store", type=Path, required=True)

    deliberate = subparsers.add_parser("deliberate")
    deliberate.add_argument("case")
    deliberate.add_argument("--question")
    deliberate.add_argument("--run-id")
    deliberate.add_argument("--approve-cost", action="store_true")
    deliberate.add_argument("--input-tokens", type=int, default=20_000)
    deliberate.add_argument("--max-results", type=int, default=10)
    deliberate.add_argument("--evidence-limit", type=int, default=20)
    _store_argument(deliberate)

    resume = subparsers.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--approve-cost", action="store_true")
    resume.add_argument("--input-tokens", type=int, default=20_000)
    _store_argument(resume)

    report = subparsers.add_parser("report")
    report.add_argument("run_id")
    report.add_argument("--format", choices=("markdown", "html", "json"), default="markdown")
    report.add_argument("--output", type=Path)
    _store_argument(report)
    for paid in (review, deliberate, resume):
        budget_arguments(paid)
    deliberate.add_argument('--fulltext-limit', type=int, default=5)
    deliberate.add_argument('--local-papers', type=Path)
    deliberate.add_argument('--approve-search', action='store_true')
    resume.add_argument('--approve-search', action='store_true')
    register(subparsers)
    return parser


def _cost_payload(estimate: CostEstimate) -> dict[str, Any]:
    age_days = (date.today() - estimate.pricing_as_of).days
    return {
        "approved": estimate.approved,
        "pricing_as_of": estimate.pricing_as_of.isoformat(),
        "pricing_stale_warning": age_days > 90,
        "call_count": estimate.call_count,
        "estimated_usd": estimate.estimated_usd,
        "hard_budget_enforced": estimate.budget_usd is not None,
        'budget_usd': estimate.budget_usd,
        "estimate_limitations": "Input/image tokens are estimates; retries, long-context tiers and billing adjustments may increase cost.",
        "lines": [line.model_dump(mode="json") for line in estimate.lines],
        "paid_calls_made": 0,
        "instruction": "--approve-cost와 --budget-usd가 필요합니다. 이미지 입력은 검증된 --image-tokens-bound도 필요합니다.",
    }


def _latest_case_payload(ledger: AppendOnlyJsonlLedger, kind: str, case_id: str, key: str):
    for record in reversed(ledger.records()):
        if record["kind"] != kind:
            continue
        payload = record["payload"]
        candidate = payload.get(key, payload)
        candidate_case = candidate.get("case_id") if isinstance(candidate, dict) else None
        if candidate_case == case_id or payload.get("case_id") == case_id:
            return candidate
    raise KeyError(f"no {kind} data found for case {case_id}")


def _latest_intake(ledger: AppendOnlyJsonlLedger, case_id: str) -> FileIntakeResult:
    files = {}
    blocks = {}
    for record in ledger.records():
        if record["kind"] != "REAL_FILES_INGESTED":
            continue
        payload = record["payload"]["intake"]
        if payload["case_id"] != case_id:
            continue
        intake = FileIntakeResult.model_validate_json(json.dumps(payload, ensure_ascii=False))
        for item in intake.files:
            artifact_id = item.artifact.artifact_id
            files[artifact_id] = item
            blocks[artifact_id] = tuple(b for b in intake.blocks if b.artifact_id == artifact_id)
    if not files:
        raise KeyError(f"no REAL_FILES_INGESTED data found for case {case_id}")
    all_blocks = tuple(block for artifact_id in sorted(files) for block in blocks[artifact_id])
    return FileIntakeResult(
        intake_id=deterministic_id("case-intake", {"case_id": case_id,
            "files": [files[key].model_dump(mode="json") for key in sorted(files)],
            "blocks": [block.model_dump(mode="json") for block in all_blocks]}),
        case_id=case_id, files=tuple(files[key] for key in sorted(files)), blocks=all_blocks,
    )


def _imaging_observations(review_id, kind, items):
    return [{**item, 'source': review_id,
             'fact_id': deterministic_id('imaging-observation',
                 {'review_id': review_id, 'kind': kind, 'index': index, 'observation': item}),
             'confirmation_scope': 'MANUAL_IMAGING_OBSERVATION_NOT_DIAGNOSIS'}
            for index, item in enumerate(items)]


def _latest_context(ledger: AppendOnlyJsonlLedger, case_id: str) -> dict[str, Any]:
    facts_by_key: dict[str, dict[str, Any]] = {}
    key_order: list[str] = []
    unstructured_facts: list[str] = []
    uncertainties: list[str] = []
    extraction_ids: list[str] = []
    extraction_calls: dict[tuple[str, str, str], dict[str, Any]] = {}
    imaging_reviews = {}
    found = False
    for record in ledger.records():
        if record['kind'] == 'IMAGING_REVIEW_WITHDRAWN' and record['payload'].get('case_id') == case_id:
            review_id = record['payload']['review_id']
            imaging_reviews = {key: value for key, value in imaging_reviews.items()
                               if value['review_id'] != review_id}
            uncertainties.append(f'Imaging review withdrawn: {review_id}; no prior measurement automatically restored.')
            continue
        if record['kind'] == 'IMAGING_REVIEW_IMPORTED' and record['payload'].get('case_id') == case_id:
            payload = record['payload']
            review = payload['review']
            imaging_reviews[(review['study_uid'], review['series_uid'])] = {
                'review_id': payload['review_id'], 'review_status': review['review_status'],
                'replaces_review_id': payload.get('replaces_review_id'),
                'amendment_reason': payload.get('amendment_reason'),
                'review_status_provenance': review['review_status_provenance'],
                'measurements': _imaging_observations(payload['review_id'], 'measurement', review['measurements']),
                'segmentations': _imaging_observations(payload['review_id'], 'segmentation', review['segmentations']),
                'conditions': review.get('conditions', {}),
                'interpretation': review['interpretation'], 'unmarked_tissue': review['unmarked_tissue'],
            }
            continue
        if record['kind'] == 'SOURCE_DECISION' and record['payload'].get('case_id') == case_id:
            decision = record['payload']
            key = decision['fact_id']
            if decision['withdraw'] or decision['value'] is None:
                facts_by_key.pop(key, None)
                uncertainties.append(f'Source review withdrawn or deferred: {key}')
            elif key in facts_by_key:
                facts_by_key[key] = {**facts_by_key[key], 'value': decision['value']}
            continue
        if record["kind"] != "HIGH_IMPACT_FACTS_CONFIRMED":
            continue
        payload = record["payload"]
        if payload.get("case_id") != case_id:
            continue
        found = True
        context = payload["context"]
        for key, value in context.get('source_decisions', {}).items():
            if value is None:
                facts_by_key.pop(key, None)
        for call in context.get("extraction_calls", []):
            extraction_calls[(call["provider"], call["provider_response_id"], call["stage"])] = call
        details = context.get("fact_details", [])
        represented = set()
        for detail in details:
            represented.add(f"{detail['fact_key']}: {detail.get('value')}")
            key = detail.get("fact_id") or deterministic_id("legacy-source-fact", {
                "key": detail["fact_key"], "value": detail.get("value"),
                "sources": sorted(detail.get("source_locators", [])),
                "scope": detail.get("scope", {}),
            })
            if key not in key_order:
                key_order.append(key)
            facts_by_key[key] = detail
        for fact in context.get("confirmed_facts", []):
            if fact not in represented and fact not in unstructured_facts:
                unstructured_facts.append(fact)
        for uncertainty in context.get("uncertainties", []):
            if uncertainty not in uncertainties:
                uncertainties.append(uncertainty)
        if context.get("extraction_id"):
            extraction_ids.append(context["extraction_id"])
    if not found and not imaging_reviews:
        raise KeyError(f"no HIGH_IMPACT_FACTS_CONFIRMED data found for case {case_id}")
    if not found:
        uncertainties.append('No source-reviewed clinical documents; imaging annotations alone are not a diagnosis.')
    details = [facts_by_key[key] for key in key_order if key in facts_by_key]
    return {
        "confirmed_facts": unstructured_facts + [
            f"{item['fact_key']}: {item.get('value')}" for item in details
        ],
        "fact_details": details,
        "uncertainties": uncertainties,
        "extraction_ids": extraction_ids,
        "extraction_calls": list(extraction_calls.values()),
        'imaging_reviews': list(imaging_reviews.values()),
    }


def _interactive_review(service: ClinicalExtractionService, extraction: ExtractionResult) -> dict[str, object]:
    decisions: dict[str, str | None] = {}
    for fact in service.pending_facts(extraction):
        prompt = (
            f"[{fact.fact_key}] 제안값={fact.proposed_value!r} / 근거={', '.join(fact.source_locators)}\n"
            f"문맥={fact.scope.model_dump(exclude_none=True)}\n"
            "문서 기재 내용 검수 (의학적 사실 확정 아님): Enter=원문 일치, 새 값=정정, ?=보류: "
        )
        answer = input(prompt).strip()
        decisions[fact.fact_id] = None if answer == "?" else answer or fact.proposed_value
        service.save_decision(extraction, fact.fact_id, decisions[fact.fact_id])
    return service.confirm(extraction, decisions)


def _render_completed(root: Path, result: TribunalResult, output: Path | None = None):
    report = ReportRenderer.build(result)
    return ReportRenderer.write(report, output or root / "reports" / result.run_id)


def _literature_retriever(limit: int):
    return OfficialLiteratureRetriever((PubMedSource(limit=limit), EuropePmcSource(limit=limit),
                                       ClinicalTrialsSource(limit=limit), OfficialAnkleGuidelineSource()))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, 'run_id', None):
        run_id(args.run_id)
    handled = execute(args)
    if handled is not None:
        code, payload = handled
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return code
    root = Path(args.store)
    ledger, compiler, projector = _components(root)
    command = args.command

    if command == "ingest":
        result = compiler.compile_json(args.case_id, args.source)
        output = {
            "case_id": args.case_id, "compile_id": result.compile_id,
            "integrity_status": result.integrity.status.value,
            "candidate_count": len(result.candidates),
            "review_queue_count": len(result.review_queue), "reused": result.reused,
            "labels": ["SYNTHETIC", "NON-CLINICAL"],
        }
    elif command == "ingest-files":
        roles = {
            path: ImageRole.REFERENCE_IMAGE_SCREENSHOT
            for path in args.paths
            if args.reference_images and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        result = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files(
            args.case, args.paths, image_roles=roles,
            password_provider=(
                (lambda path: getpass.getpass(f"PDF password ({path.name}): "))
                if args.pdf_password_prompt else None
            ),
        )
        output = {
            "case_id": result.case_id, "intake_id": result.intake_id,
            "file_count": len(result.files), "block_count": len(result.blocks),
            "reference_image_count": sum(
                item.image_role == ImageRole.REFERENCE_IMAGE_SCREENSHOT for item in result.files
            ),
            "labels": ["PERSONAL_RESEARCH", "UNTRUSTED_SOURCE_DATA"],
        }
    elif command == "review" and args.decisions:
        case_id = args.case_id or args.case
        if not case_id:
            raise ValueError("review requires CASE or --case-id")
        bundle = ReviewBundle.model_validate_json(args.decisions.read_bytes())
        requested_ids = {entry.candidate_id for entry in bundle.decisions}
        all_compiles = compiler.list_compiles(case_id)
        matches = (
            [result for result in all_compiles if result.compile_id == args.compile_id]
            if args.compile_id
            else [result for result in all_compiles if requested_ids and requested_ids.issubset(
                {candidate.candidate_id for candidate in result.review_queue}
            )]
        )
        if len(matches) != 1:
            raise ValueError("review decisions must identify exactly one compiled artifact")
        result = projector.review_and_materialize(matches[0], args.decisions)
        output = {
            "case_id": case_id, "state_id": result.state.state_id,
            "state_version": result.state.version, "claim_view_id": result.claim_view.claim_view_id,
            "claim_readiness": result.claim_view.readiness.value,
            "new_event_count": len(result.new_events), "labels": ["SYNTHETIC", "NON-CLINICAL"],
        }
    elif command == "review":
        case_id = args.case_id or args.case
        if not case_id or not args.interactive:
            raise ValueError("real-file review requires CASE --interactive")
        registry = ModelRegistry.default()
        if args.approve_cost and args.budget_usd is None:
            raise ValueError('paid calls require --budget-usd')
        try:
            intake = _latest_intake(ledger, case_id)
        except KeyError:
            if args.approve_cost:
                raise
            intake = None
        batch_count = len(extraction_batches(intake)) if intake else 1
        estimate = registry.estimate_extraction_cost(args.input_tokens, batch_count=batch_count)
        if not args.approve_cost:
            print(json.dumps({**_cost_payload(estimate), "batch_count": batch_count,
                              "estimate_scope": "all batches; completed calls reused"}, ensure_ascii=False, sort_keys=True))
            return 2
        gateways = build_live_gateways(registry)
        service = ClinicalExtractionService(root, gateways, registry)
        extraction = service.extract(intake, approved_estimate(args, estimate))
        context = _interactive_review(service, extraction)
        output = {
            "case_id": case_id, "extraction_id": extraction.extraction_id,
            "confirmed_fact_count": len(context["confirmed_facts"]),
            "uncertainty_count": len(context["uncertainties"]),
            "labels": ["PERSONAL_RESEARCH", "USER_REVIEWED"],
        }
    elif command == "analyze":
        run = ReasoningService(ledger).analyze(args.case_id, args.claim, SyntheticReasoner())
        output = run.model_dump(mode="json")
    elif command == "deliberate":
        registry = ModelRegistry.default()
        if args.approve_cost and args.budget_usd is None:
            raise ValueError('paid calls require --budget-usd')
        if args.evidence_limit < 1:
            raise ValueError("evidence-limit must be positive")
        estimate = registry.estimate_research_cost(args.input_tokens, evidence_limit=args.evidence_limit,
                                                  fulltext_limit=args.fulltext_limit)
        if not args.approve_cost:
            print(json.dumps(_cost_payload(estimate), ensure_ascii=False, sort_keys=True))
            return 2
        context = _latest_context(ledger, args.case)
        gateways = build_live_gateways(registry)
        questions = QuestionSet.for_comprehensive_review(args.case, args.question)
        result = ResearchWorkflow(root, gateways, _literature_retriever(args.max_results), registry).run(
            run_id=args.run_id, case_id=args.case, patient_context=context,
            question_set=questions, estimate=approved_estimate(args, estimate), evidence_limit=args.evidence_limit,
            fulltext_limit=args.fulltext_limit, local_papers=read_json(args.local_papers) if args.local_papers else {},
            search_approved=args.approve_search,
        )
        paths = _render_completed(root, result)
        output = {
            "case_id": args.case, "run_id": result.run_id,
            "evidence_count": len(result.evidence_cards),
            "reports": {key: str(path) for key, path in paths.items()},
            "labels": ["PERSONAL_RESEARCH", "AI_MEDICAL_DECISION_SUPPORT"],
        }
    elif command == "resume":
        run_ledger = AppendOnlyJsonlLedger(root / "runs" / args.run_id / "ledger.jsonl")
        records = run_ledger.records()
        if any(record["kind"] == "TRIBUNAL_COMPLETED" for record in records):
            paths = _render_completed(root, TribunalService.load(root, args.run_id))
            print(json.dumps({"run_id": args.run_id, "resumed": True, "paid_calls_made": 0,
                              "reports": {key: str(path) for key, path in paths.items()}}, ensure_ascii=False))
            return 0
        workflow_manifest = next((r["payload"] for r in records if r["kind"] == "WORKFLOW_CREATED"), None)
        if workflow_manifest:
            if workflow_manifest.get('prompt_version') != PROMPT_VERSION:
                raise ValueError('run policy changed; preserve old run and create a new run')
            registry = ModelRegistry.model_validate_json(json.dumps(workflow_manifest["registry"]))
            estimate = CostEstimate.model_validate_json(json.dumps(workflow_manifest["estimate"]))
            if not args.approve_cost:
                print(json.dumps(_cost_payload(estimate.model_copy(update={"approved": False})), ensure_ascii=False))
                return 2
            if estimate.budget_usd is None:
                raise ValueError('legacy run has no approved budget; create a new run')
            if args.budget_usd is not None and args.budget_usd != estimate.budget_usd:
                raise ValueError('run budget is pinned; create a new approved run')
            limit = workflow_manifest["sources"][0]["limit"]
            result = ResearchWorkflow(root, build_live_gateways(registry), _literature_retriever(limit), registry).run(
                run_id=args.run_id, case_id=workflow_manifest["case_id"],
                patient_context=workflow_manifest["patient_context"],
                question_set=QuestionSet.model_validate_json(json.dumps(workflow_manifest["question_set"])),
                estimate=estimate.approve(),
                evidence_limit=workflow_manifest["evidence_limit"],
                fulltext_limit=workflow_manifest.get('fulltext_limit', 5),
                local_papers=workflow_manifest.get('local_papers', {}), search_approved=args.approve_search,
            )
            paths = _render_completed(root, result)
            print(json.dumps({"run_id": args.run_id, "resumed": True,
                              "reports": {key: str(path) for key, path in paths.items()}}, ensure_ascii=False))
            return 0
        created = next(record for record in run_ledger.records() if record["kind"] == "RESEARCH_RUN_CREATED")
        manifest = created["payload"]
        if "configuration" not in manifest:
            raise ValueError("legacy run has no pinned configuration; create a new run")
        registry = ModelRegistry.model_validate_json(json.dumps(manifest["configuration"]["registry"]))
        estimate = registry.estimate_tribunal_cost(args.input_tokens)
        if not args.approve_cost:
            print(json.dumps(_cost_payload(estimate), ensure_ascii=False, sort_keys=True))
            return 2
        questions = QuestionSet.model_validate_json(json.dumps(manifest["question_set"], ensure_ascii=False))
        evidence = tuple(
            EvidenceCard.model_validate_json(json.dumps(item, ensure_ascii=False))
            for item in manifest["evidence_cards"]
        )
        result = TribunalService(root, build_live_gateways(registry), registry).run(
            run_id=args.run_id, case_id=manifest["case_id"], patient_context=manifest["patient_context"],
            question_set=questions, evidence_cards=evidence, estimate=estimate.approve(),
            search_queries=tuple(manifest.get("search_queries", ())),
        )
        paths = _render_completed(root, result)
        output = {"run_id": result.run_id, "resumed": True,
                  "reports": {key: str(path) for key, path in paths.items()}}
    else:
        result = TribunalService.load(root, args.run_id)
        paths = _render_completed(root, result, args.output)
        selected = paths[args.format]
        output = {"run_id": args.run_id, "format": args.format, "path": str(selected)}

    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
