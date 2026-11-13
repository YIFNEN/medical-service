"""The single-provider CLI still runs three isolated review roles."""
import json
from collections import Counter
import pytest

from medic.cli import main, _render_completed
from medic.config import ModelRegistry
from medic.blinding import comparison_input, seal_result
from medic.providers import ApprovedGateway, OpenAIGateway, RawProviderResponse, RecordedGateway
from medic.reporting import ReportRenderer
from tests import test_tribunal as fixtures
from tests.test_blind_pipeline import pipeline


def test_openai_cli_smoke_estimate_has_one_call(tmp_path, capsys):
    assert main(['providers', 'smoke', '--run-id', 'openai-check', '--store', str(tmp_path)]) == 2
    estimate = json.loads(capsys.readouterr().out)
    assert estimate['call_count'] == 1
    assert {line['provider'] for line in estimate['lines']} == {'openai'}
    assert estimate['paid_calls_made'] == 0
    assert estimate['input_scope'] == 'SYNTHETIC_ONLY'


def test_approved_openai_smoke_uses_single_gateway_and_reuses_call(tmp_path, capsys, monkeypatch):
    requests = []

    def recorded(self, **request):
        requests.append(request)
        return RawProviderResponse({'echo': 'MEDIC_SYNTHETIC_SMOKE', 'structured': True},
                                   'smoke-response', input_tokens=15, output_tokens=10)

    monkeypatch.setattr(OpenAIGateway, 'generate_raw', recorded)
    args = ['providers', 'smoke', '--run-id', 'openai-check', '--store', str(tmp_path),
            '--approve-cost', '--budget-usd', '1', '--approval-id', 'synthetic-smoke']
    for _ in range(2):
        assert main(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['status'] == 'COMPLETE'
        assert set(result['providers']) == {'openai'}
        assert result['providers']['openai']['response_id'] == 'smoke-response'
    assert len(requests) == 1


@pytest.mark.parametrize('backend', ['api', 'codex'])
def test_openai_roles_workflow_and_seal(tmp_path, backend):
    run = pipeline(tmp_path)
    registry = ModelRegistry.for_cli()
    roles = ('records', 'counterevidence', 'applicability')
    legacy_names = dict(zip(roles, fixtures.PROVIDERS))
    responses = [run.raw[name]._responses[0] for name in fixtures.PROVIDERS]
    responses += run.raw['anthropic']._responses[1:3]
    responses += [fixtures.opinion(name) for name in fixtures.PROVIDERS]
    from medic.tribunal import blind_opinion_id
    for role, name in legacy_names.items():
        batch = fixtures.peer_batch(name)
        for review, other in zip(batch['reviews'], (r for r in roles if r != role)):
            review['target_opinion_id'] = blind_opinion_id(role, f'op-{other}')
        responses.append(batch)
    responses += [dict(fixtures.revision(name), original_opinion_id=f'op-{role}')
                  for role, name in legacy_names.items()]
    responses += [fixtures.synthesis()]
    responses += [fixtures.synthesis_review(name) for name in fixtures.PROVIDERS]
    raw = RecordedGateway('openai', registry.providers[0].model, responses)
    raw.backend = backend
    run.workflow.gateways = {'openai': ApprovedGateway(raw)}
    run.workflow.registry = registry
    run.args['estimate'] = registry.estimate_research_cost(100).approve()
    if backend == 'codex':
        run.args['estimate'] = run.args['estimate'].model_copy(
            update={'backend': 'codex', 'estimated_usd': None, 'codex_calls': 18, 'transmission_approved': True})
    assert {line.provider for line in run.args['estimate'].lines} == {'openai'}
    result = run.workflow.run(**run.args)
    assert raw.call_count == 18
    assert {call.provider for call in result.calls} == {'openai'}
    assert {call.backend for call in result.calls} == {backend}
    assert tuple(op.role_id for op in result.opinions) == roles
    assert tuple(review.role_id for review in result.synthesis_reviews) == roles
    assert Counter(review.target_opinion_id for review in result.peer_reviews) == Counter(
        {op.opinion_id: 2 for op in result.opinions})
    for role, prompt in zip(roles, raw.prompts[5:8]):
        assert role in prompt.splitlines()[0]
        assert not any(op.opinion_id in prompt for op in result.opinions)
    assert run.workflow.run(**run.args) == result
    assert raw.call_count == 18
    duplicated = result.model_copy(update={'synthesis_reviews': (
        result.synthesis_reviews[0], result.synthesis_reviews[0], result.synthesis_reviews[2])})
    with pytest.raises(ValueError, match='incomplete'):
        seal_result(tmp_path, duplicated)
    _render_completed(tmp_path, result)
    assert comparison_input(tmp_path, result.run_id)['run_id'] == result.run_id
    report = ReportRenderer.markdown(ReportRenderer.build(result))
    assert '반증·대안 검토' in report
    if backend == 'codex':
        assert 'Codex 구독 실행 18건' in report
        assert all(call.estimated_cost_usd is None for call in result.calls)
        with pytest.raises(ValueError, match='backend'):
            main(['resume', result.run_id, '--backend', 'api', '--store', str(tmp_path)])


def test_openai_document_review_uses_two_source_passes_and_reconciliation(tmp_path):
    from medic.extraction import ClinicalExtractionService
    from medic.intake import FileIntakeService
    from medic.ledger import AppendOnlyJsonlLedger, ContentAddressedStore
    registry = ModelRegistry.for_cli()
    source = tmp_path / 'synthetic.txt'
    source.write_text('Synthetic record for extraction testing', encoding='utf-8')
    ledger = AppendOnlyJsonlLedger(tmp_path / 'ledger.jsonl')
    intake = FileIntakeService(ContentAddressedStore(tmp_path), ledger).ingest_files('case-1', [source])
    raw = RecordedGateway('openai', registry.providers[0].model, [
        {'provider': 'openai', 'model': 'fixture', 'provider_response_id': 'pass-one', 'facts': []},
        {'provider': 'openai', 'model': 'fixture', 'provider_response_id': 'pass-two', 'facts': []},
        {'provider_response_id': 'reconciled', 'facts': [], 'uncertainties': []},
    ])
    service = ClinicalExtractionService(tmp_path, {'openai': ApprovedGateway(raw)}, registry)
    estimate = registry.estimate_extraction_cost(100).approve()
    result = service.extract(intake, estimate)
    assert estimate.call_count == raw.call_count == 3
    assert {line.provider for line in estimate.lines} == {call.provider for call in result.calls} == {'openai'}
    assert [batch.provider_response_id for batch in result.independent_batches] == ['pass-one', 'pass-two']
    assert raw.prompts[0] != raw.prompts[1]
    assert 'pass-one' not in raw.prompts[1]
    assert 'pass-one' in raw.prompts[2] and 'pass-two' in raw.prompts[2]
    assert service.extract(intake, estimate) == result
    assert raw.call_count == 3
