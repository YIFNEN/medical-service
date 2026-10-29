from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


from .model_base import StrictFrozenModel
from .questions import QuestionClaim


ANKLE_REVIEW_DOMAINS = (
    "POST_TRAUMATIC_ANKLE_ARTHRITIS",
    "UNION_DELAYED_UNION_NONUNION",
    "MALUNION_AND_ALIGNMENT",
    "SYNDESMOSIS",
    "HARDWARE",
    "SYMPTOMS_FUNCTION_ALTERNATIVES",
    "TESTS_REHABILITATION_AND_TREATMENT_OPTIONS",
)


class QuestionSet(StrictFrozenModel):
    question_set_id: str
    case_id: str
    domains: tuple[str, ...]
    custom_question: str | None = None

    @staticmethod
    def _expected_id(case_id: str, custom_question: str | None) -> str:
        payload = {
            "case_id": case_id,
            "domains": ANKLE_REVIEW_DOMAINS,
            "custom_question": custom_question,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        return f"question-set-{digest}"

    @model_validator(mode="after")
    def require_fixed_domains_and_id(self) -> "QuestionSet":
        if self.domains != ANKLE_REVIEW_DOMAINS:
            raise ValueError("question set must contain the fixed seven ankle domains in order")
        if self.question_set_id != self._expected_id(self.case_id, self.custom_question):
            raise ValueError("question_set_id does not match the canonical question payload")
        return self

    @classmethod
    def for_comprehensive_review(
        cls, case_id: str, custom_question: str | None = None
    ) -> "QuestionSet":
        return cls(
            question_set_id=cls._expected_id(case_id, custom_question),
            case_id=case_id,
            domains=ANKLE_REVIEW_DOMAINS,
            custom_question=custom_question,
        )


class EvidenceFieldSupport(StrictFrozenModel):
    field: Literal["study_design", "population", "intervention", "comparator", "findings", "effect_estimates", "limitations"]
    value: str
    quote: str = Field(min_length=1)
    source_start: int | None = None
    source_end: int | None = None
    source_locator: str | None = None
    source_sha256: str | None = None


class SourcePassage(StrictFrozenModel):
    locator: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class EvidenceCard(StrictFrozenModel):
    evidence_id: str
    source: Literal["PUBMED", "EUROPE_PMC", "CLINICAL_TRIALS", "GUIDELINE"]
    title: str
    url: HttpUrl
    retrieved_at: datetime
    study_design: str
    population: str
    comparator: str | None = None
    findings: tuple[str, ...]
    effect_estimates: tuple[str, ...] = ()
    limitations: tuple[str, ...]
    applicability: tuple[str, ...] = ()
    pmid: str | None = None
    doi: str | None = None
    nct_id: str | None = None
    intervention: str | None = None
    source_text: str = ""
    access_level: Literal["ABSTRACT", "FULL_TEXT", "WEBPAGE", "REGISTRY", "CATALOG", "METADATA", "LEGACY_UNSPECIFIED"] = "LEGACY_UNSPECIFIED"
    field_support: tuple[EvidenceFieldSupport, ...] = ()
    appraisal_status: Literal["UNAPPRAISED", "APPRAISED", "NO_CONTENT", "LEGACY_UNSPECIFIED"] = "LEGACY_UNSPECIFIED"
    missing_fields: tuple[str, ...] = ()
    pmcid: str | None = None
    decision_issue_ids: tuple[str, ...] = ()
    screening_status: Literal['UNSCREENED', 'INCLUDE', 'EXCLUDE', 'UNCERTAIN'] = 'UNSCREENED'
    screening_reason: str = ''
    priority_score: int = 0
    fulltext_required: bool = False
    fulltext_reasons: tuple[str, ...] = ()
    fulltext_status: str = 'NOT_REQUESTED'
    analysis_scope: str = 'UNREVIEWED'
    source_sha256: str | None = None
    source_passages: tuple[SourcePassage, ...] = ()
    unresolved_source_locators: tuple[str, ...] = ()
    analyzed_locators: tuple[str, ...] = ()
    fulltext_url: str | None = None
    publication_status: Literal['NOT_CHECKED', 'SOURCE_CHECKED_NO_NOTICE', 'CORRECTION_REQUIRES_REVIEW',
                                'EXPRESSION_OF_CONCERN', 'RETRACTED'] = 'NOT_CHECKED'
    publication_updates: tuple[dict, ...] = ()
    publication_notice_reviewed: bool = False
    study_id: str | None = None
    study_link_status: Literal['UNVERIFIED', 'USER_REVIEWED'] = 'UNVERIFIED'
    included_study_ids: tuple[str, ...] = ()


class FactScope(StrictFrozenModel):
    event_date: str | None = None
    report_date: str | None = None
    issue_date: str | None = None
    institution: str | None = None
    anatomical_site: str | None = None
    laterality: str | None = None
    modality: str | None = None
    assertion: Literal["AFFIRMED", "SUSPECTED", "NEGATED", "ALTERNATIVE", "UNKNOWN"] = "UNKNOWN"
    source_kind: Literal["RADIOLOGY_REPORT", "CLINICIAN_OPINION", "PATIENT_REPORT", "OPERATIVE_RECORD", "UNKNOWN"] = "UNKNOWN"


class ConfirmedFactDetail(StrictFrozenModel):
    fact_key: str
    value: str
    source_locators: tuple[str, ...] = Field(min_length=1)
    user_confirmed: bool
    fact_id: str | None = None
    scope: FactScope = Field(default_factory=FactScope)
    conflict: bool = False
    confirmation_scope: Literal["SOURCE_TRANSCRIPTION_ONLY", "UNREVIEWED_EXTRACTION", "LEGACY_UNSPECIFIED"] = "LEGACY_UNSPECIFIED"


class DomainAssessment(StrictFrozenModel):
    domain: str
    status: Literal["ANALYZED", "DEFERRED"]
    explanation: str = Field(min_length=1)
    hypotheses: tuple[str, ...]
    supporting_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    counterevidence: tuple[str, ...]
    uncertainties: tuple[str, ...]
    conditional_actions: tuple[str, ...]
    decision_changers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_domain(self):
        if self.domain not in ANKLE_REVIEW_DOMAINS:
            raise ValueError("unknown ankle domain")
        if self.status == "ANALYZED" and not self.hypotheses:
            raise ValueError("analyzed domain requires hypotheses")
        return self


class IndependentOpinion(StrictFrozenModel):
    opinion_id: str
    provider: str
    model: str
    provider_response_id: str
    hypotheses: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    missing_information: tuple[str, ...]
    action_ranking: tuple[str, ...]
    reasoning_summary: str
    claim_abstentions: tuple[str, ...] = ()
    domain_assessments: tuple[DomainAssessment, ...] = ()


class ReviewVerdict(StrEnum):
    SUPPORT = "SUPPORT"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    OPPOSE = "OPPOSE"


class PeerReview(StrictFrozenModel):
    review_id: str
    reviewer_provider: str
    reviewer_model: str
    target_opinion_id: str
    verdict: ReviewVerdict
    strengths: tuple[str, ...] = ()
    omissions: tuple[str, ...] = ()
    evidence_errors: tuple[str, ...] = ()
    applicability_errors: tuple[str, ...] = ()
    required_changes: tuple[str, ...] = ()
    citation_errors: tuple[str, ...] = ()
    evidence_contradictions: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class PeerReviewBatch(StrictFrozenModel):
    reviewer_provider: str
    reviewer_model: str
    provider_response_id: str
    reviews: tuple[PeerReview, ...] = Field(min_length=2, max_length=2)


class RevisionDisposition(StrEnum):
    MAINTAIN = "MAINTAIN"
    MODIFY = "MODIFY"
    WITHDRAW = "WITHDRAW"


class RevisedOpinion(StrictFrozenModel):
    revised_opinion_id: str
    original_opinion_id: str
    provider: str
    model: str
    disposition: RevisionDisposition
    response_to_reviews: tuple[str, ...]
    hypotheses: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    missing_information: tuple[str, ...]
    action_ranking: tuple[str, ...]
    reasoning_summary: str
    domain_assessments: tuple[DomainAssessment, ...] = ()


class ConditionalAction(StrictFrozenModel):
    rank: int = Field(ge=1)
    action: str
    prerequisites: tuple[str, ...]
    benefits: tuple[str, ...]
    downsides: tuple[str, ...]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    question_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    decision_mode: Literal['OPTIONS', 'SINGLE_BEST'] = 'OPTIONS'


class ProposedSynthesis(StrictFrozenModel):
    synthesis_id: str
    provider: str
    model: str
    agreements: tuple[str, ...]
    disagreements: tuple[str, ...]
    conditional_action_ranking: tuple[str, ...]
    decision_changers: tuple[str, ...]
    conditional_actions: tuple[ConditionalAction, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...]
    question_claims: tuple[QuestionClaim, ...] = ()

    @model_validator(mode="after")
    def require_ranked_matching_actions(self) -> "ProposedSynthesis":
        ranks = tuple(action.rank for action in self.conditional_actions)
        if ranks != tuple(range(1, len(self.conditional_actions) + 1)):
            raise ValueError("conditional action ranks must be unique, ordered and contiguous from 1")
        if tuple(action.action for action in self.conditional_actions) != self.conditional_action_ranking:
            raise ValueError("conditional actions must exactly match conditional_action_ranking")
        return self


class ClaimQuote(StrictFrozenModel):
    evidence_id: str
    quote: str = Field(min_length=1)


class ClaimReview(StrictFrozenModel):
    claim_id: str
    verdict: Literal["SUPPORTED", "UNCERTAIN", "UNSUPPORTED", "CONTRADICTED"]
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]
    patient_fact_ids: tuple[str, ...] = ()
    source_quotes: tuple[ClaimQuote, ...]
    requires_fulltext_verification: bool = False
    required_evidence_fields: tuple[Literal['study_design', 'population', 'intervention', 'comparator',
                                          'findings', 'effect_estimates', 'limitations'], ...] = ()


class WithheldClaim(StrictFrozenModel):
    claim_id: str
    location: str
    original_text: str
    reasons: tuple[str, ...]
    status: Literal['MISSING_DATA', 'CONFLICT', 'OUT_OF_SCOPE', 'SYSTEM_FAILURE'] = 'MISSING_DATA'


class SynthesisReview(StrictFrozenModel):
    synthesis_review_id: str
    reviewer_provider: str
    reviewer_model: str
    synthesis_id: str
    verdict: ReviewVerdict
    supported_points: tuple[str, ...] = ()
    objections: tuple[str, ...] = ()
    required_changes: tuple[str, ...] = ()
    citation_errors: tuple[str, ...] = ()
    evidence_contradictions: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    claim_reviews: tuple[ClaimReview, ...] = ()


class ProviderCallRecord(StrictFrozenModel):
    stage: str
    provider: str
    model: str
    provider_response_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class MedicalDecisionReport(StrictFrozenModel):
    report_id: str
    run_id: str
    case_id: str
    question_set: QuestionSet
    data_cutoff: AwareDatetime
    confirmed_facts: tuple[str, ...]
    confirmed_fact_details: tuple[ConfirmedFactDetail, ...]
    uncertainties: tuple[str, ...]
    opinions: tuple[IndependentOpinion, ...]
    peer_reviews: tuple[PeerReview, ...]
    revised_opinions: tuple[RevisedOpinion, ...]
    synthesis: ProposedSynthesis
    synthesis_reviews: tuple[SynthesisReview, ...]
    evidence_cards: tuple[EvidenceCard, ...]
    calls: tuple[ProviderCallRecord, ...]
    pricing_as_of: date
    approved_cost_upper_bound_usd: float = Field(ge=0, allow_inf_nan=False)
    search_queries: tuple[str, ...] = ()
    labels: tuple[str, ...] = (
        "PERSONAL_RESEARCH",
        "AI_MEDICAL_DECISION_SUPPORT",
        "NOT_AUTOMATED_EXECUTION",
    )
    degraded: bool = False
    completed_at: AwareDatetime | None = None
    prompt_version: str = "medic-agent-first/1.0"
    withheld_claims: tuple[WithheldClaim, ...] = ()
    original_synthesis: ProposedSynthesis | None = None
    research_trace: dict = Field(default_factory=dict)
    imaging_reviews: tuple[dict, ...] = ()
    question_trace: dict = Field(default_factory=dict)
    case_context: dict = Field(default_factory=dict)
    analysis_input: dict = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def require_research_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        required = {
            "PERSONAL_RESEARCH",
            "AI_MEDICAL_DECISION_SUPPORT",
            "NOT_AUTOMATED_EXECUTION",
        }
        if not required.issubset(value):
            raise ValueError("research report labels are incomplete")
        return value

    @model_validator(mode="after")
    def require_canonical_confirmed_fact_coverage(self) -> "MedicalDecisionReport":
        keys = tuple(detail.fact_id or json.dumps({"key": detail.fact_key, "value": detail.value,
                     "sources": sorted(detail.source_locators)}, sort_keys=True)
                     for detail in self.confirmed_fact_details)
        if len(keys) != len(set(keys)):
            raise ValueError("confirmed source fact identities must be unique")
        canonical = tuple(
            f"{detail.fact_key}: {detail.value}" for detail in self.confirmed_fact_details
        )
        if self.confirmed_facts != canonical:
            raise ValueError("confirmed_facts must exactly match confirmed_fact_details in order")
        return self


JsonObject = dict[str, Any]
