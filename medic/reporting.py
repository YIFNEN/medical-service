from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import markdown as markdown_library

from .research_contracts import ConfirmedFactDetail, MedicalDecisionReport
from .tribunal import TribunalResult


_MARKDOWN_META = "\\`*_{}[]()#+-.!|>"


def _safe_text(value: object) -> str:
    """Render untrusted dynamic data as inert single-line Markdown text."""
    escaped = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    for character in _MARKDOWN_META:
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def _safe_url(value: object) -> str:
    return html.escape(str(value), quote=True).replace("(", "%28").replace(")", "%29")

def _safe_filename(value: object) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return candidate or hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]



def _inline(values: Iterable[str], empty: str = "없음") -> str:
    items = tuple(values)
    return "; ".join(_safe_text(value) for value in items) if items else empty


def _source_locators(values: Iterable[str]) -> str:
    locators = tuple(values)
    if not locators:
        return "없음"
    return "; ".join(
        "`" + html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ").replace("`", "&#96;") + "`"
        for value in locators
    )


def _detail(label: str, values: Iterable[str]) -> str:
    return f"{label}:\n\n{_items(values, empty='없음')}\n"

def _fact_details(values: Iterable[ConfirmedFactDetail]) -> str:
    details = tuple(values)
    if not details:
        return "- 구조화된 원문 위치 기록 없음"
    lines: list[str] = []
    for detail in details:
        lines.extend((
            f"- {_safe_text(detail.fact_key)}: {_safe_text(detail.value)}",
            f"  - 원문 위치: {_source_locators(detail.source_locators)}",
            f"  - 사용자 확인: {'예' if detail.user_confirmed else '아니오'}",
            f"  - 확인 범위: {_safe_text(detail.confirmation_scope)} (문서 기재 내용 확인과 의학적 확정은 다름)",
            f"  - 날짜·부위·검사·출처: {_safe_text(json.dumps(detail.scope.model_dump(exclude_none=True), ensure_ascii=False))}",
            f"  - 상충 소견: {'예' if detail.conflict else '아니오'}",
        ))
    return "\n".join(lines)


def _conditional_action_details(values) -> str:
    actions = tuple(values)
    if not actions:
        return "- 구조화된 조건부 행동 기록 없음"
    lines: list[str] = []
    for action in sorted(actions, key=lambda item: item.rank):
        lines.extend((
            f"### {action.rank}. {_safe_text(action.action)}",
            "",
            _detail("전제조건", action.prerequisites),
            _detail("장점", action.benefits),
            _detail("단점", action.downsides),
            _detail("근거 ID", action.evidence_ids),
            "",
        ))
    return "\n".join(lines).rstrip()

def _items(values: Iterable[str], empty: str = "기록 없음") -> str:
    items = tuple(values)
    return "\n".join(f"- {_safe_text(value)}" for value in items) if items else f"- {empty}"


def _strict_strings(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must contain only strings")
    return tuple(value)


def _canonical_fact(detail: ConfirmedFactDetail) -> str:
    return f"{detail.fact_key}: {detail.value}"


class ReportRenderer:
    @staticmethod
    def build(result: TribunalResult, *, degraded: bool = False) -> MedicalDecisionReport:
        context = result.patient_context
        payload = f"{result.run_id}:{result.created_at.isoformat()}"
        report_id = f"medical-report-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"
        raw_facts = _strict_strings(context.get("confirmed_facts", ()), "confirmed_facts")
        uncertainties = _strict_strings(context.get("uncertainties", ()), "uncertainties")
        raw_details = context.get("fact_details", ())
        if not isinstance(raw_details, (list, tuple)):
            raise TypeError("fact_details must be a list or tuple")
        fact_details: list[ConfirmedFactDetail] = []
        for item in raw_details:
            if not isinstance(item, dict):
                raise TypeError("fact_details must contain only objects")
            locators = item.get("source_locators", ())
            if not isinstance(locators, (list, tuple)):
                raise TypeError("source_locators must be a list or tuple of strings")
            fact_details.append(ConfirmedFactDetail.model_validate({
                **item,
                "source_locators": tuple(locators),
            }))
        confirmed_fact_details = tuple(fact_details)
        confirmed_facts = tuple(_canonical_fact(detail) for detail in confirmed_fact_details)
        if raw_facts != confirmed_facts:
            raise ValueError("confirmed_facts must exactly match fact_details in order")
        return MedicalDecisionReport(
            report_id=report_id,
            run_id=result.run_id,
            case_id=result.case_id,
            question_set=result.question_set,
            data_cutoff=result.data_cutoff or result.created_at,
            completed_at=result.created_at,
            prompt_version=result.prompt_version,
            withheld_claims=result.withheld_claims,
            original_synthesis=result.original_synthesis,
            confirmed_facts=confirmed_facts,
            uncertainties=uncertainties,
            confirmed_fact_details=confirmed_fact_details,
            opinions=result.opinions,
            peer_reviews=result.peer_reviews,
            revised_opinions=result.revised_opinions,
            synthesis=result.synthesis,
            synthesis_reviews=result.synthesis_reviews,
            evidence_cards=result.evidence_cards,
            calls=result.calls,
            pricing_as_of=result.pricing_as_of,
            approved_cost_upper_bound_usd=result.approved_cost_upper_bound_usd,
            search_queries=result.search_queries,
            degraded=degraded,
            research_trace=context.get('literature_review', {}),
            imaging_reviews=tuple(context.get('imaging_reviews', ())),
        )

    @staticmethod
    def markdown(report: MedicalDecisionReport) -> str:
        lines = [
            "# 개인 발목 의학적 판단 지원 보고서",
            "",
            "> 이 보고서는 PERSONAL_RESEARCH / AI_MEDICAL_DECISION_SUPPORT 결과이며, "
            "처방·수술·약물 변경 등 의료행위를 자동 실행하지 않습니다.",
            "",
            "## 1. 분석 범위와 자료 cutoff",
            "",
            f"- Case: {_safe_text(report.case_id)}",
            f"- Run: {_safe_text(report.run_id)}",
            f"- 자료 cutoff: {_safe_text(report.data_cutoff.isoformat())}",
            f"- 추가 질문: {_safe_text(report.question_set.custom_question or '없음')}",
            f"- Degraded mode: {'예' if report.degraded else '아니오'}",
            f"- Labels: {_inline(report.labels)}",
            "",
            "## 2. 확인된 환자 사실과 원문 위치",
            "",
            _fact_details(report.confirmed_fact_details),
            "",
            "## 3. 누락·충돌·불확실한 사실",
            "",
            _items(report.uncertainties),
            '### 문헌 검증 범위와 미해결 쟁점',
            _safe_text(json.dumps(report.research_trace, ensure_ascii=False)),
            '### 영상 계측·수동 표시 (임상 정확도 미검증)',
            _items(json.dumps(item, ensure_ascii=False) for item in report.imaging_reviews),
            "",
            "## 4. 일곱 영역별 독립 의견",
            "",
        ]
        for opinion in report.opinions:
            for domain in opinion.domain_assessments:
                lines.extend([
                    f"### {_safe_text(opinion.provider)}: {_safe_text(domain.domain)} ({domain.status})",
                    _safe_text(domain.explanation), _detail("후보 가설", domain.hypotheses),
                    _detail("지지 사실 ID", domain.supporting_fact_ids), _detail("문헌 근거 ID", domain.evidence_ids),
                    _detail("반증", domain.counterevidence), _detail("불확실성", domain.uncertainties),
                    _detail("조건부 행동", domain.conditional_actions), _detail("판단을 바꿀 자료", domain.decision_changers), "",
                ])
            lines.extend(
                [
                    f"### {_safe_text(opinion.provider)} / {_safe_text(opinion.model)}",
                    "",
                    _safe_text(opinion.reasoning_summary),
                    "",
                    "가설:",
                    "",
                    _items(opinion.hypotheses),
                    "",
                    "추가 필요 정보:",
                    "",
                    _items(opinion.missing_information),
                    "",
                    "판단 보류:",
                    "",
                    _items(opinion.claim_abstentions),
                    "",
                    "행동순위:",
                    "",
                    _items(opinion.action_ranking),
                    "",
                    _detail("근거 ID", opinion.evidence_ids),
                    "",
                ]
            )
        lines.extend(["## 5. 각 의견에 대한 다른 Agent들의 평가", ""])
        for review in report.peer_reviews:
            lines.extend(
                [
                    f"### {_safe_text(review.reviewer_provider)} → {_safe_text(review.target_opinion_id)}: {_safe_text(review.verdict.value)}",
                    "",
                    _detail("강점", review.strengths),
                    _detail("누락", review.omissions),
                    _detail("근거 오류", review.evidence_errors),
                    _detail("개인 적용 오류", review.applicability_errors),
                    _detail("필수 변경", review.required_changes),
                    _detail("인용 오류", review.citation_errors),
                    _detail("근거 모순", review.evidence_contradictions),
                    _detail("근거 ID", review.evidence_ids),
                    "",
                ]
            )
        lines.extend(["## 6. 비평 후 수정·철회된 의견", ""])
        for revised in report.revised_opinions:
            lines.extend(
                [
                    f"### {_safe_text(revised.provider)}: {_safe_text(revised.disposition.value)}",
                    "",
                    _safe_text(revised.reasoning_summary),
                    "",
                    "비평 응답:",
                    "",
                    _items(revised.response_to_reviews),
                    "",
                    "수정 가설:",
                    "",
                    _items(revised.hypotheses),
                    "",
                    _detail("추가 필요 정보", revised.missing_information),
                    "수정 행동순위:",
                    "",
                    _items(revised.action_ranking),
                    "",
                    _detail("근거 ID", revised.evidence_ids),
                    "",
                ]
            )
        lines.extend(
            [
                "## 7. 공통점과 핵심 이견",
                "",
                f"- 합성 제공자: {_safe_text(report.synthesis.provider)} / {_safe_text(report.synthesis.model)}",
                _detail("합성 근거 ID", report.synthesis.evidence_ids),
                "",
                "### 공통점",
                "",
                _items(report.synthesis.agreements),
                "",
                "### 핵심 이견",
                "",
                _items(report.synthesis.disagreements),
                "",
                "### 합성안 재평가",
                "",
            ]
        )
        for review in report.synthesis_reviews:
            lines.extend((
                f"#### {_safe_text(review.reviewer_provider)}: {_safe_text(review.verdict.value)}",
                "",
                _detail("지지 지점", review.supported_points),
                _detail("이의", review.objections),
                _detail("필수 변경", review.required_changes),
                _detail("인용 오류", review.citation_errors),
                _detail("근거 모순", review.evidence_contradictions),
                _detail("근거 ID", review.evidence_ids),
                "",
            ))
        for claim in report.withheld_claims:
            lines.extend([f"### 보류된 주장: {_safe_text(claim.claim_id)}",
                          f"검증 전 원문 (결론으로 채택하지 않음): {_safe_text(claim.original_text)}",
                          _detail("보류 이유", claim.reasons), ""])
        lines.extend(
            [
                "",
                "## 8. 조건부 검사·치료 행동순위",
                "",
                _items(report.synthesis.conditional_action_ranking),
                "",
                "## 9. 각 순위의 전제조건과 예상 장단점",
                "",
                "",
                _conditional_action_details(report.synthesis.conditional_actions),
                "## 10. 결론을 바꿀 추가자료와 정보가치",
                "",
                _items(report.synthesis.decision_changers),
                "",
                "## 11. 최신 근거와 직접 링크",
                "",
            ]
        )
        for card in report.evidence_cards:
            identifiers = tuple(
                item for item in (f"PMID {card.pmid}" if card.pmid else "", f"DOI {card.doi}" if card.doi else "", f"NCT {card.nct_id}" if card.nct_id else "") if item
            )
            lines.extend(
                [
                    f"- [{_safe_text(card.title)}]({_safe_url(card.url)}) — {_safe_text(card.study_design)}; {_inline(identifiers, _safe_text(card.source))}",
                    f"  - 출처/수집시각: {_safe_text(card.source)} / {_safe_text(card.retrieved_at.isoformat())}",
                    f"  - 대상군: {_safe_text(card.population)}",
                    f"  - 접근 범위/추출 상태: {_safe_text(card.access_level)} / {_safe_text(card.appraisal_status)}",
                    f"  - 실제 분석 범위/전문 확보: {_safe_text(card.analysis_scope)} / {_safe_text(card.fulltext_status)}",
                    f"  - 읽지 못한 원문 위치: {_source_locators(card.unresolved_source_locators)}",
                    f"  - 선별 이유/쟁점: {_safe_text(card.screening_reason)} / {_inline(card.decision_issue_ids)}",
                    f"  - 전문 검증 필요 사유: {_inline(card.fulltext_reasons)}",
                    f"  - 중재: {_safe_text(card.intervention or '기재 없음')}",
                    f"  - 비교군: {_safe_text(card.comparator or '없음')}",
                    f"  - 핵심: {_inline(card.findings)}",
                    f"  - 효과 추정치: {_inline(card.effect_estimates)}",
                    f"  - 한계: {_inline(card.limitations)}",
                    f"  - 개인 적용성: {_inline(card.applicability)}",
                    f"  - 미기재 항목: {_inline(card.missing_fields)}",
                ]
            )
            for support in card.field_support:
                lines.append(f"  - {_safe_text(support.field)} 근거 [{support.source_start}:{support.source_end}] "
                             f"{_safe_text(support.source_locator or 'text')} / {_safe_text(support.source_sha256 or '')}: {_safe_text(support.quote)}")
        lines.extend([
            "",
            "## 12. 모델·프롬프트·검색식·비용·run 계보",
            "",
            f"- 가격 기준일: {_safe_text(report.pricing_as_of.isoformat())}",
            f"- 승인 당시 비용 추정: USD {report.approved_cost_upper_bound_usd:.6f} (예산 상한 집행 아님)",
            f"- 기록된 추출·검색계획·근거추출·Tribunal 사용량 기반 비용 추정 (재사용한 과거 추출 포함): USD {sum(call.estimated_cost_usd for call in report.calls):.6f}",
            f"- 프롬프트 정책: {_safe_text(report.prompt_version)}",
            f"- 분석 완료시각: {_safe_text(report.completed_at)}",
            "- 검색식:",
            _items(report.search_queries),
            "",
        ])
        if report.calls:
            for call in report.calls:
                lines.append(
                    f"- {_safe_text(call.stage)}: {_safe_text(call.provider)}/{_safe_text(call.model)}, response {_safe_text(call.provider_response_id)}, "
                    f"tokens {call.input_tokens}+{call.output_tokens}"
                )
        else:
            lines.append("- 저장된 provider 호출 기록 없음")
        lines.extend(["", f"- 질문 세트: {_safe_text(report.question_set.question_set_id)}", f"- 보고서: {_safe_text(report.report_id)}", ""])
        return "\n".join(lines)

    @classmethod
    def write(cls, report: MedicalDecisionReport, output_dir: Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_filename(report.run_id)
        markdown_path = output_dir / f"{stem}.md"
        html_path = output_dir / f"{stem}.html"
        json_path = output_dir / f"{stem}.json"
        rendered = cls.markdown(report)
        markdown_path.write_text(rendered, encoding="utf-8")
        body = markdown_library.markdown(rendered, extensions=("tables",))
        html_path.write_text(
            "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<title>개인 발목 의학적 판단 지원 보고서</title></head><body>"
            + body
            + "</body></html>",
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"markdown": markdown_path, "html": html_path, "json": json_path}
