import json
from contextlib import redirect_stdout
from io import StringIO

import pytest

from medic.cli import main
from medic.config import ModelRegistry


def test_cli_pilot_overrides_share_cost_settings(tmp_path):
    output = StringIO()
    with redirect_stdout(output):
        code = main(['deliberate', 'synthetic', '--preset', 'pilot',
                     '--evidence-limit', '8', '--fulltext-limit', '3', '--store', str(tmp_path)])
    result = json.loads(output.getvalue())
    assert code == 2
    assert result['paid_calls_made'] == 0
    assert result['processing_settings']['evidence_limit'] == 8
    assert result['processing_settings']['fulltext_limit'] == 3
    assert result['processing_settings']['screening_limit'] == 20
    assert result['call_count'] == 16 + 8 + 1 + 2 + 36


def test_resolve_precedence_and_reject_unknown_unsafe_settings(tmp_path):
    from medic.settings import resolve_settings
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps({'evidence_limit': 6, 'fulltext_limit': 2}), encoding='utf-8')
    resolved = resolve_settings('pilot', config, {'evidence_limit': 8, 'fulltext_limit': None})
    assert resolved.evidence_limit == 8
    assert resolved.fulltext_limit == 2
    assert resolved.max_results == 5
    config.write_text('{"skip_citation_check": true}', encoding='utf-8')
    with pytest.raises(ValueError):
        resolve_settings('pilot', config)
    with pytest.raises(ValueError):
        resolve_settings(overrides={'screening_batch_size': 0})


def test_estimate_uses_screen_and_chunk_settings():
    from medic.settings import ProcessingSettings
    settings = ProcessingSettings(screening_limit=21, screening_batch_size=4,
                                  evidence_limit=3, fulltext_limit=1, fulltext_max_chunks=2)
    estimate = ModelRegistry.default().estimate_research_cost(100, settings=settings, question_count=7)
    assert estimate.call_count == 16 + 3 + 1 + 6 + 2
    assert estimate.question_count == 7
    assert estimate.processing_settings == settings.model_dump(mode='json')


def test_standalone_literature_stages_accept_the_same_processing_settings():
    from medic.cli import _parser
    from medic.settings import settings_from_args
    args = _parser().parse_args(['literature', 'appraise', '--cards', 'synthetic.json',
                                 '--preset', 'pilot', '--fulltext-chunk-chars', '9000'])
    settings = settings_from_args(args)
    assert settings.fulltext_chunk_chars == 9000
    assert settings.fulltext_limit == 1


def test_default_store_and_network_free_doctor_do_not_print_secret(tmp_path, monkeypatch):
    from medic.cli import _parser
    monkeypatch.setenv('OPENAI_API_KEY', 'DO-NOT-OUTPUT-THIS-SECRET')
    assert str(_parser().parse_args(['deliberate', 'c']).store) == '.medic'
    output = StringIO()
    with redirect_stdout(output):
        assert main(['doctor', '--store', str(tmp_path)]) in (0, 2)
    text = output.getvalue()
    result = json.loads(text)
    assert result['network_calls_made'] == 0
    assert result['providers'][0]['key_available'] is True
    assert 'DO-NOT-OUTPUT-THIS-SECRET' not in text
    assert not list(tmp_path.iterdir())


def test_doctor_rejects_a_file_as_store_or_store_parent(tmp_path):
    from medic.doctor import inspect_environment
    path = tmp_path / 'file'
    path.write_text('synthetic')
    assert not inspect_environment(path)['offline_ready']
    assert not inspect_environment(path / 'nested')['offline_ready']


def test_resume_preserves_legacy_workflow_manifest(tmp_path):
    from unittest.mock import patch
    from medic.ledger import AppendOnlyJsonlLedger
    from medic.research_contracts import QuestionSet
    from medic.workflow import ResearchWorkflow, ResearchRunFailed
    from tests.test_tribunal import gateways
    from medic.evidence import OfficialLiteratureRetriever
    raw, approved = gateways()
    workflow = ResearchWorkflow(tmp_path, approved, OfficialLiteratureRetriever(()))
    args = dict(case_id='case-1', run_id='legacy', patient_context={},
                question_set=QuestionSet.for_comprehensive_review('case-1'),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
    with patch('medic.workflow.LiteratureSearchPlanner.plan', side_effect=ValueError('sentinel')):
        with pytest.raises(ResearchRunFailed):
            workflow.run(**args)
    saved = AppendOnlyJsonlLedger(tmp_path / 'runs' / 'legacy' / 'ledger.jsonl').records()[0]['payload']
    saved.pop('processing_settings')
    root2 = tmp_path / 'old-store'
    ledger = AppendOnlyJsonlLedger(root2 / 'runs' / 'legacy' / 'ledger.jsonl')
    ledger.append_once('WORKFLOW_CREATED', 'legacy', saved)
    workflow2 = ResearchWorkflow(root2, approved, OfficialLiteratureRetriever(()))
    with patch('medic.workflow.LiteratureSearchPlanner.plan', side_effect=ValueError('sentinel')):
        with pytest.raises(ResearchRunFailed) as raised:
            workflow2.run(**args)
    assert str(raised.value.__cause__) == 'sentinel'
    assert ledger.records()[0]['payload'] == saved


def test_legacy_processing_settings_restore_actual_search_source_limit(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from medic.evidence import OfficialLiteratureRetriever
    from medic.ledger import AppendOnlyJsonlLedger
    from medic.research_contracts import QuestionSet
    from medic.workflow import ResearchWorkflow, ResearchRunFailed
    from tests.test_tribunal import gateways
    class Source:
        limit = 5
    _, approved = gateways()
    args = dict(case_id='case-1', run_id='legacy-five', patient_context={}, search_approved=True,
        question_set=QuestionSet.for_comprehensive_review('case-1'),
        estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
    first = ResearchWorkflow(tmp_path, approved, OfficialLiteratureRetriever((Source(),)))
    with patch('medic.workflow.LiteratureSearchPlanner.plan', side_effect=ValueError('stop')):
        with pytest.raises(ResearchRunFailed):
            first.run(**args)
    saved = AppendOnlyJsonlLedger(tmp_path / 'runs' / args['run_id'] / 'ledger.jsonl').records()[0]['payload']
    saved.pop('processing_settings')
    root2 = tmp_path / 'old'
    AppendOnlyJsonlLedger(root2 / 'runs' / args['run_id'] / 'ledger.jsonl').append_once('WORKFLOW_CREATED', args['run_id'], saved)
    second = ResearchWorkflow(root2, approved, OfficialLiteratureRetriever((Source(),)))
    with patch('medic.workflow.LiteratureSearchPlanner.plan', return_value=SimpleNamespace(queries=())), \
         patch('medic.workflow.LiteratureScreener') as screener:
        screener.return_value.issues.side_effect = ValueError('stop')
        with pytest.raises(ResearchRunFailed):
            second.run(**args)
        assert screener.call_args.args[-1].max_results == 5
