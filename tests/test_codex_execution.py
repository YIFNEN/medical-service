import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from medic.cli import main
from medic.config import ModelRegistry
from medic.provider_checks import SmokeReply


def test_codex_smoke_cli_preview_run_and_reuse(tmp_path, monkeypatch, capsys):
    from medic.codex_gateway import CodexGateway
    from medic.providers import RawProviderResponse

    calls = []

    def generate(self, **kwargs):
        calls.append(kwargs)
        return RawProviderResponse({'echo': 'MEDIC_SYNTHETIC_SMOKE', 'structured': True},
                                   'test-turn', input_tokens=20, output_tokens=10)

    monkeypatch.setattr(CodexGateway, 'generate_raw', generate)
    args = ['providers', 'smoke', '--backend', 'codex', '--run-id', 'test-codex', '--store', str(tmp_path)]
    assert main(args) == 2
    preview = json.loads(capsys.readouterr().out)
    assert preview['backend'] == 'codex'
    assert preview['call_count'] == 1
    assert preview['estimated_usd'] is None
    assert not calls
    for _ in range(2):
        assert main([*args, '--codex-calls', '1']) == 0
        report = json.loads(capsys.readouterr().out)
        assert report['backend'] == 'codex'
        assert report['providers']['openai']['input_tokens'] == 20
    assert len(calls) == 1


def test_codex_image_smoke_checks_attachment_and_rejects_wrong_color(tmp_path, monkeypatch, capsys):
    from PIL import Image
    from medic.codex_gateway import CodexGateway
    from medic.providers import RawProviderResponse

    calls = []

    def generate(self, *, image_paths, **kwargs):
        with Image.open(image_paths[0]) as image:
            assert image.size == (256, 256)
            color = {(255, 0, 0): 'red', (0, 128, 0): 'green', (0, 0, 255): 'blue'}[image.getpixel((128, 128))]
            assert image.getpixel((0, 0)) == (255, 255, 255)
        calls.append(color)
        return RawProviderResponse({'square_color': color if len(calls) == 1 else 'white'}, 'test-turn')

    monkeypatch.setattr(CodexGateway, 'generate_raw', generate)
    args = ['providers', 'smoke', '--backend', 'codex', '--synthetic-image', '--codex-calls', '1',
            '--store', str(tmp_path), '--run-id']
    for _ in range(2):
        assert main([*args, 'smoke-1']) == 0
        assert json.loads(capsys.readouterr().out)['status'] == 'PASSED'
    assert len(calls) == 1
    with pytest.raises(ValueError, match='image color'):
        main([*args, 'smoke-2'])


@pytest.mark.parametrize('unexpected_tool', [False, True])
def test_codex_cli_command_uses_chatgpt_auth_stdin_and_validates_events(monkeypatch, unexpected_tool):
    from medic.codex_gateway import CodexGateway

    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-key')
    monkeypatch.setenv('CODEX_API_KEY', 'fixture-key')
    monkeypatch.setattr('medic.codex_gateway.shutil.which', lambda _: 'fixture-codex.exe')
    prompt = 'Synthetic request with a private-looking marker'

    def run(command, **kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env'] and 'CODEX_API_KEY' not in kwargs['env']
        if command[1:3] == ['login', 'status']:
            return SimpleNamespace(returncode=0, stdout='Logged in using ChatGPT', stderr='')
        assert '--ignore-user-config' in command and '--ephemeral' in command
        assert 'read-only' in command and '--output-schema' in command
        assert not any(prompt in arg for arg in command)
        assert prompt in kwargs['input']
        Path(command[command.index('--output-last-message') + 1]).write_text(
            '{"echo":"MEDIC_SYNTHETIC_SMOKE","structured":true}', encoding='utf-8')
        events = [{'type': 'thread.started', 'thread_id': 'example-thread'}]
        if unexpected_tool:
            events.append({'type': 'item.completed', 'item': {'type': 'command_execution'}})
        events.append({'type': 'turn.completed', 'usage': {'input_tokens': 100, 'output_tokens': 10}})
        return SimpleNamespace(returncode=0, stdout='\n'.join(map(json.dumps, events)), stderr='')

    monkeypatch.setattr('medic.codex_gateway.subprocess.run', run)
    kwargs = dict(stage='check', prompt=prompt, json_schema=SmokeReply.model_json_schema(),
                  max_output_tokens=100)
    if unexpected_tool:
        with pytest.raises(RuntimeError, match='unexpected tool'):
            CodexGateway().generate_raw(**kwargs)
    else:
        result = CodexGateway().generate_raw(**kwargs)
        assert (result.response_id, result.input_tokens, result.output_tokens) == ('example-thread', 100, 10)


def test_codex_failed_attempt_counts_and_personal_requires_consent(tmp_path, monkeypatch):
    from medic.call_cache import cached_generate
    from medic.codex_gateway import CodexGateway
    from medic.ledger import AppendOnlyJsonlLedger
    from medic.providers import ApprovedGateway

    calls = []

    def fail(self, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('mock interruption')

    monkeypatch.setattr(CodexGateway, 'generate_raw', fail)
    gateway = ApprovedGateway(CodexGateway())
    ledger = AppendOnlyJsonlLedger(tmp_path / 'ledger.jsonl')
    estimate = ModelRegistry.for_cli().estimate_tribunal_cost(100).approve().model_copy(
        update={'backend': 'codex', 'codex_calls': 1})
    kwargs = dict(stage='check', prompt='synthetic', output_model=SmokeReply, max_output_tokens=100)
    with pytest.raises(PermissionError, match='transmission'):
        cached_generate(ledger, gateway, scope='test', estimate=estimate, **kwargs)
    assert not calls
    estimate = estimate.model_copy(update={'transmission_approved': True})
    with pytest.raises(RuntimeError, match='mock interruption'):
        cached_generate(ledger, gateway, scope='test', estimate=estimate, **kwargs)
    with pytest.raises(RuntimeError, match='limit'):
        cached_generate(ledger, gateway, scope='test', estimate=estimate, **kwargs)
    assert len(calls) == 1
