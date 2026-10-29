"""Bounded M6 verification with synthetic inputs and recorded provider responses.

These checks establish input separation and artifact integrity, not semantic
de-identification, exhaustive literature coverage, or clinical accuracy.
"""
import copy
import json
import socket
import unicodedata
from collections import Counter
from types import SimpleNamespace

import pytest

from medic.blinding import assert_no_leak, comparison_input, create_blind_input, seal_result
from medic.cli import _render_completed
from medic.config import ModelRegistry
from medic.evidence import OfficialLiteratureRetriever
from medic.ledger import AppendOnlyJsonlLedger, canonical_json_bytes, sha256_digest
from medic.question_runtime import available_references
from medic.questions import evaluate_questions
from medic.reporting import ReportRenderer
from medic.research_contracts import QuestionSet
from medic.tribunal import TribunalService
from medic.workflow import ResearchRunFailed, ResearchWorkflow
import tests.test_blinding as blind_fixtures
import tests.test_tribunal as tribunal_fixtures
from tests.test_workflow_resume import plan


HIDDEN = 'SEALED CLINICIAN PROGNOSIS'
KOREAN = '의료진의 가려진 장기 예후 판단입니다'
NESTED_NOTE = 'EXCLUDED IMAGING COMMENTARY'


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('M6 recorded verification must never use the network')

    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


class RecordedSource:
    def __init__(self):
        self.card = tribunal_fixtures.TribunalTests().evidence()
        self.queries = []
        self.last_search = {'scope': 'synthetic recorded source'}

    def search(self, query):
        self.queries.append(query)
        return (self.card,)


class NoFullText:
    def fetch(self, card):
        raise AssertionError('This bounded fixture must not fetch full text')


def source_context():
    context = blind_fixtures.context()
    context['typed_questions'] = [
        {'question_id': identifier, 'type': 'CURRENT_STATE', 'text': 'Reviewed current state?',
         'target': 'ankle', 'assessment_time': 'reviewed time', 'allowed_fact_ids': [fact_id]}
        for identifier, fact_id in [('visible-question', 'direct'), ('withheld-question', 'doctor')]
    ]
    context['question_evaluations'] = [r.model_dump(mode='json') for r in evaluate_questions(
        context['typed_questions'], available=available_references(context))]
    context['question_version'] = {'event_id': 'questions-reviewed', 'version': 1}
    context['imaging_reviews'] = [{
        'review_id': 'reviewed-image', 'review_status': 'USER_REVIEWED',
        'measurements': [{'fact_id': 'measured', 'measurement_id': 'm', 'kind': 'line',
                          'value': 5, 'units': 'mm',
                          'metadata': {'laterality': 'RIGHT', 'method': 'reviewed line',
                                       'acquisition': {'modality': 'CT', 'protocol': 'synthetic protocol',
                                                       'clinician_interpretation': NESTED_NOTE},
                                       'clinician_interpretation': {'nested': [HIDDEN]}},
                          'axes': {'x': [1, 0, 0], 'y': [0, 1, 0], 'source': 'reviewed axes',
                                   'clinician_interpretation': NESTED_NOTE}}],
        'conditions': {'modality': 'CT', 'geometry': {'shape': [2, 2, 2],
                                                   'clinician_interpretation': NESTED_NOTE}},
        'segmentations': [], 'clinician_interpretation': HIDDEN,
    }]
    return context


def pipeline(root, run_id='blind-pipeline', context=None):
    context = source_context() if context is None else context
    questions = QuestionSet.for_comprehensive_review('case-1')
    approval = blind_fixtures.approval()
    approval['imaging_review_ids'] = ['reviewed-image']
    visible = create_blind_input(root, run_id, 'case-1', context,
                                 questions.model_dump(mode='json'), approval)
    raw, approved = tribunal_fixtures.gateways()
    for provider, gateway in raw.items():
        gateway._responses.insert(0, plan(provider))
    raw['anthropic']._responses[1:1] = [
        {'issues': [{'issue_id': 'issue', 'question': 'General explanation',
                     'required_evidence': ['review'], 'missing_information': [],
                     'decision_changers': [], 'triggers': [],
                     'question_ids': ['visible-question', 'withheld-question']}]},
        {'decisions': [{'evidence_id': 'ev-1', 'verdict': 'INCLUDE',
                        'reason': 'Synthetic relevant review', 'issue_ids': ['issue'],
                        'impact': 1, 'applicability': 1, 'methods_needed': False}]},
    ]
    source = RecordedSource()
    workflow = ResearchWorkflow(root, approved, OfficialLiteratureRetriever((source,)))
    args = dict(run_id=run_id, case_id='case-1', patient_context=visible,
                question_set=questions, search_approved=True, resolver=NoFullText(),
                estimate=ModelRegistry.default().estimate_deliberation_cost(100).approve())
    return SimpleNamespace(workflow=workflow, args=args, raw=raw, source=source,
                           visible=visible, original=context,
                           ledger=AppendOnlyJsonlLedger(root / 'runs' / run_id / 'ledger.jsonl'))


def prompts(run):
    return [prompt for gateway in run.raw.values() for prompt in gateway.prompts]


def test_recorded_blind_planning_screen_tribunal_and_reports_exclude_nested_clinician_text(tmp_path):
    run = pipeline(tmp_path)
    result = run.workflow.run(**run.args)
    paths = _render_completed(tmp_path, result)
    seal = next(r['payload'] for r in run.ledger.records() if r['kind'] == 'BLIND_RESULT_SEALED')
    report = ReportRenderer.build(result)

    stages = Counter(record['payload']['stage'] for record in run.ledger.records()
                     if record['kind'] == 'PROVIDER_CALL_COMPLETED')
    assert stages == Counter({'LITERATURE_SEARCH_PLANNING': 3, 'DECISION_ISSUES': 1,
                             'LITERATURE_SCREENING': 1, 'ANALYST_COMPLETED': 3,
                             'PEER_REVIEW_COMPLETED': 3, 'REVISION_COMPLETED': 3,
                             'SYNTHESIS_COMPLETED': 1, 'SYNTHESIS_REVIEW_COMPLETED': 3})
    assert len(prompts(run)) == len(result.calls) == 18
    assert run.source.queries == list(result.search_queries) == ['ankle fracture union']
    assert len(result.opinions) == len(result.synthesis_reviews) == 3
    assert len(result.peer_reviews) == 6
    assert result.evidence_cards[0].screening_status == 'INCLUDE'
    assert report.analysis_input['mode'] == 'CLINICIAN_BLIND'
    assert seal['result_sha256'] == sha256_digest(canonical_json_bytes(result.model_dump(mode='json')))
    surfaces = [*prompts(run), *run.source.queries, result.model_dump_json(), report.model_dump_json(),
                *(path.read_text(encoding='utf-8') for path in paths.values())]
    for surface in surfaces:
        assert HIDDEN not in surface
        assert NESTED_NOTE not in surface
        assert 'unreviewed-page.png' not in surface
    assert all('walking limited' in gateway.prompts[0] for gateway in run.raw.values())
    assert report.imaging_reviews[0]['measurements'][0]['value'] == 5
    assert report.imaging_reviews[0]['measurements'][0]['metadata']['acquisition'] == {
        'modality': 'CT', 'protocol': 'synthetic protocol'}

    # A completed run must reload without spending more calls or querying sources.
    assert run.workflow.run(**run.args) == result
    assert len(prompts(run)) == 18
    assert run.source.queries == ['ankle fracture union']


def test_visibility_change_recomputes_readiness_in_every_patient_prompt_and_report(tmp_path):
    run = pipeline(tmp_path)
    assert [r['status'] for r in run.original['question_evaluations']] == ['ANSWER', 'ANSWER']
    expected = run.visible['question_evaluations']
    assert [r['status'] for r in expected] == ['ANSWER', 'CONFLICT']
    assert expected[1]['reasons'] == ['unknown_fact_reference:doctor']
    result = run.workflow.run(**run.args)
    seal_result(tmp_path, result)
    report = ReportRenderer.build(result)
    assert result.patient_context['question_evaluations'] == expected
    assert [r['status'] for r in report.question_trace['evaluations']] == ['ANSWER', 'CONFLICT']
    patient_prompts = [prompt for prompt in prompts(run) if '"question_evaluations"' in prompt]
    assert len(patient_prompts) == 15  # every stage carrying the reviewed patient context
    for prompt in patient_prompts:
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{'):])
        patient = next(payload[key] for key in ('patient_context', 'patient') if key in payload)
        assert patient['question_evaluations'] == expected


@pytest.mark.parametrize('phrase', [HIDDEN, KOREAN, '의료진의 "가려진" 장기 예후 판단입니다',
                                  '의료진의\n가려진\t장기 예후 판단입니다'])
@pytest.mark.parametrize('surface', ['question', 'nested-metadata'])
def test_escaped_withheld_input_rejected_before_any_provider_or_snapshot(tmp_path, phrase, surface):
    context = source_context()
    context['fact_details'][1]['value'] = phrase
    questions = QuestionSet.for_comprehensive_review('case-1').model_dump(mode='json')
    encoded = json.dumps({'comment': unicodedata.normalize('NFD', phrase)}, ensure_ascii=True)
    if surface == 'question':
        questions['custom_question'] = encoded
    else:
        context['imaging_reviews'][0]['measurements'][0]['metadata']['method'] = encoded
    approval = blind_fixtures.approval()
    approval['imaging_review_ids'] = ['reviewed-image']
    with pytest.raises(ValueError, match='withheld'):
        create_blind_input(tmp_path, 'rejected', 'case-1', context, questions, approval)
    assert not AppendOnlyJsonlLedger(tmp_path / 'runs' / 'rejected' / 'ledger.jsonl').records()
    assert not (tmp_path / 'withheld-inputs').exists()


def test_late_escaped_korean_metadata_stopped_before_first_provider_call(tmp_path):
    context = source_context()
    context['fact_details'][1]['value'] = KOREAN
    run = pipeline(tmp_path, context=context)
    # Exercise the paid boundary independently of the initial snapshot check.
    run.visible['imaging_reviews'][0]['measurements'][0]['metadata']['method'] = json.dumps(
        {'nested': [KOREAN]}, ensure_ascii=True)
    with pytest.raises(ResearchRunFailed) as failure:
        run.workflow.run(**run.args)
    assert isinstance(failure.value.__cause__, ValueError)
    assert 'withheld' in str(failure.value.__cause__)
    assert not prompts(run)
    assert not run.source.queries


def unicode_backslash_chain(layers):
    # Each pass reveals one more backslash escape; the final pass yields "A".
    return r'\u005c' + ('u005c' * (layers - 2)) + 'u0041'


def test_json_escape_decoding_accepts_exactly_sixteen_layers(tmp_path):
    run = pipeline(tmp_path)
    assert_no_leak(tmp_path, run.args['run_id'], unicode_backslash_chain(16))
    assert not prompts(run)


@pytest.mark.parametrize('layers', [17, 64])
def test_deep_json_escape_chain_rejected_before_snapshot(tmp_path, layers):
    context = source_context()
    context['imaging_reviews'][0]['measurements'][0]['metadata']['method'] = unicode_backslash_chain(layers)
    with pytest.raises(ValueError, match='16 JSON escape decoding layers'):
        pipeline(tmp_path, context=context)
    ledger = AppendOnlyJsonlLedger(tmp_path / 'runs' / 'blind-pipeline' / 'ledger.jsonl')
    assert not ledger.records()
    assert not (tmp_path / 'withheld-inputs').exists()


@pytest.mark.parametrize('layers', [17, 64])
def test_deep_json_escape_chain_fails_closed_before_first_provider(tmp_path, layers):
    run = pipeline(tmp_path)
    run.visible['imaging_reviews'][0]['measurements'][0]['metadata']['method'] = unicode_backslash_chain(layers)
    with pytest.raises(ResearchRunFailed) as failure:
        run.workflow.run(**run.args)
    assert isinstance(failure.value.__cause__, ValueError)
    assert '16 JSON escape decoding layers' in str(failure.value.__cause__)
    assert not prompts(run)
    assert not run.source.queries
    assert not any(r['kind'] in ('TRIBUNAL_COMPLETED', 'BLIND_RESULT_SEALED')
                   for r in run.ledger.records())


@pytest.mark.parametrize('surface,expected_calls', [('query', 3), ('issue', 4),
                                                   ('search-metadata', 5), ('opinion', 6)])
def test_contamination_is_stopped_before_next_external_boundary(tmp_path, surface, expected_calls):
    context = source_context()
    context['fact_details'][1]['value'] = KOREAN
    run = pipeline(tmp_path, context=context)
    encoded = json.dumps({'nested': [KOREAN]}, ensure_ascii=True)
    if surface == 'query':
        run.raw['openai']._responses[0]['queries'] = [KOREAN]
    elif surface == 'issue':
        run.raw['anthropic']._responses[1]['issues'][0]['question'] = encoded
    elif surface == 'search-metadata':
        run.source.last_search['metadata'] = {'comment': encoded}
    else:
        run.raw['openai']._responses[1]['reasoning_summary'] = encoded
    with pytest.raises(ResearchRunFailed) as failure:
        run.workflow.run(**run.args)
    assert isinstance(failure.value.__cause__, ValueError)
    assert 'withheld' in str(failure.value.__cause__)
    assert len(prompts(run)) == expected_calls
    assert run.source.queries == ([] if surface == 'query' else ['ankle fracture union'])
    assert not any(r['kind'] in ('TRIBUNAL_COMPLETED', 'BLIND_RESULT_SEALED')
                   for r in run.ledger.records())
    assert not (tmp_path / 'reports').exists()


@pytest.mark.parametrize('surface', ['case-mismatch', 'provider-metadata'])
def test_report_path_requires_same_case_seal_and_checks_output_only_metadata(tmp_path, surface):
    context = source_context()
    context['fact_details'][1]['value'] = KOREAN
    run = pipeline(tmp_path, context=context)
    if surface == 'provider-metadata':
        # Response metadata is retained for audit but is absent from later prompts.
        run.raw['openai']._responses[0]['_response_id'] = json.dumps({'nested': [KOREAN]}, ensure_ascii=True)
    result = run.workflow.run(**run.args)
    if surface == 'case-mismatch':
        result = result.model_copy(update={'case_id': 'another-case'})
    assert len(prompts(run)) == 18
    with pytest.raises(ValueError, match='pinned|withheld'):
        _render_completed(tmp_path, result)
    assert not any(r['kind'] == 'BLIND_RESULT_SEALED' for r in run.ledger.records())
    assert not (tmp_path / 'reports').exists()


def test_comparison_requires_same_case_complete_seal_and_preserves_raw_report_bytes(tmp_path):
    run = pipeline(tmp_path)
    result = run.workflow.run(**run.args)
    with pytest.raises(ValueError, match='sealed'):
        comparison_input(tmp_path, result.run_id)
    for update, message in [({'case_id': 'another-case'}, 'pinned'),
                            ({'synthesis_reviews': result.synthesis_reviews[:-1]}, 'incomplete'),
                            ({'created_at': result.created_at.replace(year=2000)}, 'persisted')]:
        with pytest.raises(ValueError, match=message):
            seal_result(tmp_path, result.model_copy(update=update))
    assert not any(r['kind'] == 'BLIND_RESULT_SEALED' for r in run.ledger.records())
    seal = seal_result(tmp_path, result)
    paths = ReportRenderer.write(ReportRenderer.build(result), tmp_path / 'reports' / result.run_id)
    before = {path: path.read_bytes() for path in [run.ledger.path, *paths.values()]}
    original_result = copy.deepcopy(result.model_dump(mode='json'))

    comparison = comparison_input(tmp_path, result.run_id)
    assert comparison['case_id'] == result.case_id
    assert comparison['seal'] == seal
    assert [f['fact_id'] for f in comparison['clinician_facts']] == ['doctor']
    assert comparison['clinician_facts'][0]['value'] == HIDDEN
    assert comparison['comparison_scope'] == 'OPINION_COMPARISON_NOT_CLINICAL_ACCURACY'
    assert comparison['agreement_accuracy_score'] is None
    from medic.claim_grounding import synthesis_claims
    assert comparison['claims'] == list(synthesis_claims(result.original_synthesis))
    comparison['claims'][0]['text'] = HIDDEN
    comparison['clinician_facts'][0]['value'] = 'modified comparison only'
    assert comparison_input(tmp_path, result.run_id)['clinician_facts'][0]['value'] == HIDDEN
    assert result.model_dump(mode='json') == original_result
    assert TribunalService.load(tmp_path, result.run_id) == result
    assert seal_result(tmp_path, result) == seal
    assert {path: path.read_bytes() for path in before} == before
    assert sum(r['kind'] == 'BLIND_RESULT_SEALED' for r in run.ledger.records()) == 1
