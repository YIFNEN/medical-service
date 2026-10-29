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

    def run(self, *, case_id, patient_context, question_set, estimate, run_id=None, evidence_limit=None,
            fulltext_limit=None, local_papers=None, resolver=None, search_approved=False, settings=None):
        if not estimate.approved:
            raise PermissionError("explicit cost approval is required before workflow execution")
        from .settings import ProcessingSettings
        requested_settings = settings
        values = (settings or ProcessingSettings()).model_dump(mode='json')
        values.update({k: v for k, v in {'evidence_limit': evidence_limit,
                                       'fulltext_limit': fulltext_limit}.items() if v is not None})
        settings = ProcessingSettings.model_validate(values)
        evidence_limit, fulltext_limit = settings.evidence_limit, settings.fulltext_limit
        run_id = run_id or f"research-run-{uuid.uuid4().hex}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
            raise ValueError("invalid run_id")
        ledger = AppendOnlyJsonlLedger(self.root / "runs" / run_id / "ledger.jsonl")
        saved = next((r["payload"] for r in ledger.records() if r["kind"] == "WORKFLOW_CREATED"), None)
        if saved is not None and 'processing_settings' not in saved:
            search_limit = next((s['limit'] for s in saved.get('sources', ()) if s.get('limit') is not None), 10)
            legacy = ProcessingSettings(max_results=search_limit, evidence_limit=saved['evidence_limit'],
                                        fulltext_limit=saved.get('fulltext_limit', 5))
            if requested_settings is not None and settings != legacy:
                raise ValueError('legacy run settings are pinned; create a new run')
            if evidence_limit != legacy.evidence_limit or fulltext_limit != legacy.fulltext_limit:
                raise ValueError('legacy evidence/fulltext limits are pinned; create a new run')
            settings = legacy
        if estimate.processing_settings is not None and estimate.processing_settings != settings.model_dump(mode='json'):
            raise ValueError('approved estimate and execution settings differ; create a new approved run')
        manifest = {
            "case_id": case_id, "run_id": run_id, "patient_context": patient_context,
            "question_set": question_set.model_dump(mode="json"),
            "registry": self.registry.model_dump(mode="json"), "prompt_version": PROMPT_VERSION,
            "evidence_limit": evidence_limit,
            "fulltext_limit": fulltext_limit, 'local_papers': local_papers or {},
            'screening_limit': settings.screening_limit,
            'processing_settings': settings.model_dump(mode='json'),
            "models": {name: g.model for name, g in self.gateways.items()},
            "sources": [{"type": type(s).__name__, "limit": getattr(s, "limit", None)} for s in self.retriever.sources],
            "estimate": saved["estimate"] if saved else estimate.model_dump(mode="json"),
            "data_cutoff": saved["data_cutoff"] if saved else datetime.now(UTC).isoformat(),
        }
        if saved is not None and 'processing_settings' not in saved:
            # Historical manifests stay byte-equivalent; they used fixed batching/chunking.
            manifest.pop('processing_settings')
        ledger.append_once("WORKFLOW_CREATED", run_id, manifest)
        approved = CostEstimate.model_validate_json(json.dumps(manifest["estimate"]))
        try:
            if any(r["kind"] == "TRIBUNAL_COMPLETED" for r in ledger.records()):
                return TribunalService.load(self.root, run_id)
            plans = LiteratureSearchPlanner(self.gateways, self.registry).plan(
                question_set, patient_context, approved, ledger=ledger)
            if patient_context.get('analysis_input', {}).get('mode') == 'CLINICIAN_BLIND':
                from .blinding import assert_no_leak
                assert_no_leak(self.root, run_id, '\n'.join(plans.queries))
            search_key = deterministic_id('search-review', {'queries': list(plans.queries)})
            ledger.append_once('SEARCH_REVIEW_REQUIRED', search_key, {'queries': list(plans.queries)})
            if not any(r['kind'] == 'SEARCH_APPROVED' and r['natural_key'] == search_key for r in ledger.records()):
                if not search_approved:
                    raise PermissionError('review literature search-plan, then resume with --approve-search')
                ledger.append_once('SEARCH_APPROVED', search_key, {'queries': list(plans.queries)})
            evidence = self.retriever.search(plans.queries, ledger=ledger)
            appraiser = EvidenceAppraiser(self.gateways["anthropic"], ledger, self.registry, settings)
            screener = LiteratureScreener(self.gateways['anthropic'], ledger, self.registry, settings)
            issues = screener.issues(question_set, patient_context, approved)
            candidates = sorted(evidence.cards, key=lambda c: (not (c.source == 'GUIDELINE' or
                any(term in c.study_design.lower() for term in ('systematic', 'meta-analysis'))), c.evidence_id))
            screened = screener.screen(tuple(candidates[:settings.screening_limit]), issues, patient_context, approved)
            selected, pending = select_evidence(screened, evidence_limit)
            pending += tuple(c.evidence_id for c in candidates[settings.screening_limit:])
            ledger.append_once("EVIDENCE_SELECTION", run_id, {
                "selected_ids": [card.evidence_id for card in selected],
                "excluded_ids": [card.evidence_id for card in screened if card.screening_status == 'EXCLUDE'],
                'pending_ids': list(pending), 'decisions': [c.model_dump(mode='json') for c in screened],
                "limit": evidence_limit, "selection_order": "decision impact, applicability, methods; not agreement",
            })
            resolver = resolver or FullTextResolver(ledger=ledger, local_papers=local_papers)
            enriched, requested, fetch_requests = [], 0, 0
            for card in selected:
                if card.fulltext_required or card.access_level == 'FULL_TEXT':
                    if requested < fulltext_limit:
                        requested += 1
                        if card.access_level != 'FULL_TEXT':
                            card = resolver.fetch(card)
                            fetch_requests += 1
                        else:
                            # Acquired source processing is reusable; applicability appraisal is per-run.
                            card = card.model_copy(update={'appraisal_status': 'UNAPPRAISED', 'analysis_scope': 'UNREVIEWED'})
                    else:
                        updates = {'fulltext_status': 'PENDING_LIMIT'}
                        if card.access_level == 'FULL_TEXT':
                            updates.update(appraisal_status='UNAPPRAISED', analysis_scope='UNREVIEWED')
                        card = card.model_copy(update=updates)
                enriched.append(card)
            analysis_context = {**patient_context, 'decision_issues': [i.model_dump(mode='json') for i in issues]}
            appraised = tuple(appraiser.appraise(card, analysis_context, approved)
                              if card.appraisal_status == "UNAPPRAISED" and not
                              (card.access_level == 'FULL_TEXT' and card.fulltext_status == 'PENDING_LIMIT')
                              else card for card in enriched)
            analysis_context['literature_review'] = {
                'processing_settings': settings.model_dump(mode='json'),
                'counts': {'selected_cards': len(selected),
                           'unique_studies': None, 'unique_studies_status': 'NOT_YET_DEDUPLICATED_BY_STUDY',
                           'fulltext_targets': requested, 'fulltext_fetch_requests': fetch_requests,
                           'fulltext_attempted': len({r['payload']['evidence_id'] for r in ledger.records()
                                                      if r['kind'] == 'FULLTEXT_SOURCE_ATTEMPT'}),
                           'fulltext_acquired': sum(c.access_level == 'FULL_TEXT' for c in appraised),
                           'fulltext_analyzed': sum(c.analysis_scope in ('FULL_TEXT', 'PARTIAL_FULL_TEXT') for c in appraised)},
                'pending_ids': list(pending), 'fulltext_limit': fulltext_limit,
                'pending_fulltext_ids': [c.evidence_id for c in appraised if c.fulltext_status == 'PENDING_LIMIT'],
                'unresolved_evidence_ids': [c.evidence_id for c in appraised if c.fulltext_required and c.analysis_scope != 'FULL_TEXT'],
                'issues': [i.model_dump(mode='json') for i in issues],
                'search_completeness': 'bounded search; not an exhaustive systematic review',
                'search_metadata': [r['payload'].get('search_metadata', {}) for r in ledger.records()
                                    if r['kind'] == 'SOURCE_SEARCH_COMPLETED'],
                'budget_usd': approved.budget_usd,
            }
            from .literature_integrity import study_counts
            analysis_context['literature_review']['counts'].update(study_counts(appraised))
            usable = tuple(card for card in appraised if card.appraisal_status not in {"UNAPPRAISED", "NO_CONTENT"})
            from .question_runtime import scope_questions
            analysis_context['scoped_questions'] = scope_questions(patient_context, usable, issues)
            if not usable:
                raise ValueError("no appraised evidence with retrieved content")
            return TribunalService(self.root, self.gateways, self.registry).run(
                run_id=run_id, case_id=case_id, patient_context=analysis_context,
                question_set=question_set, evidence_cards=usable, estimate=approved,
                search_queries=plans.queries, data_cutoff=datetime.fromisoformat(manifest["data_cutoff"]),
            )
        except Exception as exc:
            raise ResearchRunFailed(f"run {run_id} paused at {type(exc).__name__}; use medic resume {run_id}") from exc
