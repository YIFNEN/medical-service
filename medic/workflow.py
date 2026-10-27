import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import CostEstimate, ModelRegistry
from .evidence import EvidenceBundle, LiteratureSearchPlanner
from .evidence_appraisal import EvidenceAppraiser
from .ledger import AppendOnlyJsonlLedger, deterministic_id
from .tribunal import PROMPT_VERSION, TribunalService
from .literature import LiteratureScreener, FullTextResolver, select_evidence


class ResearchRunFailed(RuntimeError):
    pass


class ResearchWorkflow:
    def __init__(self, root: Path, gateways, retriever, registry=None):
        self.root, self.gateways, self.retriever = Path(root), gateways, retriever
        self.registry = registry or ModelRegistry.default()

    def run(self, *, case_id, patient_context, question_set, estimate, run_id=None, evidence_limit=20,
            fulltext_limit=5, local_papers=None, resolver=None, search_approved=False):
        if not estimate.approved:
            raise PermissionError("explicit cost approval is required before workflow execution")
        if fulltext_limit < 0 or evidence_limit < 1:
            raise ValueError('invalid evidence/full-text limits')
        run_id = run_id or f"research-run-{uuid.uuid4().hex}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
            raise ValueError("invalid run_id")
        ledger = AppendOnlyJsonlLedger(self.root / "runs" / run_id / "ledger.jsonl")
        saved = next((r["payload"] for r in ledger.records() if r["kind"] == "WORKFLOW_CREATED"), None)
        manifest = {
            "case_id": case_id, "run_id": run_id, "patient_context": patient_context,
            "question_set": question_set.model_dump(mode="json"),
            "registry": self.registry.model_dump(mode="json"), "prompt_version": PROMPT_VERSION,
            "evidence_limit": evidence_limit,
            "fulltext_limit": fulltext_limit, 'local_papers': local_papers or {},
            'screening_limit': 200,
            "models": {name: g.model for name, g in self.gateways.items()},
            "sources": [{"type": type(s).__name__, "limit": getattr(s, "limit", None)} for s in self.retriever.sources],
            "estimate": saved["estimate"] if saved else estimate.model_dump(mode="json"),
            "data_cutoff": saved["data_cutoff"] if saved else datetime.now(UTC).isoformat(),
        }
        ledger.append_once("WORKFLOW_CREATED", run_id, manifest)
        approved = CostEstimate.model_validate_json(json.dumps(manifest["estimate"]))
        try:
            if any(r["kind"] == "TRIBUNAL_COMPLETED" for r in ledger.records()):
                return TribunalService.load(self.root, run_id)
            plans = LiteratureSearchPlanner(self.gateways, self.registry).plan(
                question_set, patient_context, approved, ledger=ledger)
            search_key = deterministic_id('search-review', {'queries': list(plans.queries)})
            ledger.append_once('SEARCH_REVIEW_REQUIRED', search_key, {'queries': list(plans.queries)})
            if not any(r['kind'] == 'SEARCH_APPROVED' and r['natural_key'] == search_key for r in ledger.records()):
                if not search_approved:
                    raise PermissionError('review literature search-plan, then resume with --approve-search')
                ledger.append_once('SEARCH_APPROVED', search_key, {'queries': list(plans.queries)})
            evidence = self.retriever.search(plans.queries, ledger=ledger)
            appraiser = EvidenceAppraiser(self.gateways["anthropic"], ledger, self.registry)
            screener = LiteratureScreener(self.gateways['anthropic'], ledger, self.registry)
            issues = screener.issues(question_set, patient_context, approved)
            candidates = sorted(evidence.cards, key=lambda c: (not (c.source == 'GUIDELINE' or
                any(term in c.study_design.lower() for term in ('systematic', 'meta-analysis'))), c.evidence_id))
            screened = screener.screen(tuple(candidates[:200]), issues, patient_context, approved)
            selected, pending = select_evidence(screened, evidence_limit)
            pending += tuple(c.evidence_id for c in candidates[200:])
            ledger.append_once("EVIDENCE_SELECTION", run_id, {
                "selected_ids": [card.evidence_id for card in selected],
                "excluded_ids": [card.evidence_id for card in screened if card.screening_status == 'EXCLUDE'],
                'pending_ids': list(pending), 'decisions': [c.model_dump(mode='json') for c in screened],
                "limit": evidence_limit, "selection_order": "decision impact, applicability, methods; not agreement",
            })
            resolver = resolver or FullTextResolver(ledger=ledger, local_papers=local_papers)
            enriched, requested = [], 0
            for card in selected:
                if card.fulltext_required:
                    if requested < fulltext_limit:
                        card = resolver.fetch(card)
                        requested += 1
                    else:
                        card = card.model_copy(update={'fulltext_status': 'PENDING_LIMIT'})
                enriched.append(card)
            analysis_context = {**patient_context, 'decision_issues': [i.model_dump(mode='json') for i in issues]}
            appraised = tuple(appraiser.appraise(card, analysis_context, approved)
                              if card.appraisal_status == "UNAPPRAISED" else card for card in enriched)
            analysis_context['literature_review'] = {
                'pending_ids': list(pending), 'fulltext_limit': fulltext_limit,
                'unresolved_evidence_ids': [c.evidence_id for c in appraised if c.fulltext_required and c.analysis_scope != 'FULL_TEXT'],
                'issues': [i.model_dump(mode='json') for i in issues],
                'search_completeness': 'bounded search; not an exhaustive systematic review',
                'search_metadata': [r['payload'].get('search_metadata', {}) for r in ledger.records()
                                    if r['kind'] == 'SOURCE_SEARCH_COMPLETED'],
                'budget_usd': approved.budget_usd,
            }
            usable = tuple(card for card in appraised if card.appraisal_status not in {"UNAPPRAISED", "NO_CONTENT"})
            if not usable:
                raise ValueError("no appraised evidence with retrieved content")
            return TribunalService(self.root, self.gateways, self.registry).run(
                run_id=run_id, case_id=case_id, patient_context=analysis_context,
                question_set=question_set, evidence_cards=usable, estimate=approved,
                search_queries=plans.queries, data_cutoff=datetime.fromisoformat(manifest["data_cutoff"]),
            )
        except Exception as exc:
            raise ResearchRunFailed(f"run {run_id} paused at {type(exc).__name__}; use medic resume {run_id}") from exc
